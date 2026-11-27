"""Resolve multi-entity projections with a role-sensitive schema graph."""

from collections import deque
import re

from semplan import Plan, choose, col, ident, words


def graph(schema):
    edges = {
        t: [(u, k, r, False) for u, k, r, _ in items]
        for t, items in schema.edges.items()
    }
    targets = {}
    for t, pk in schema.pk.items():
        if len(pk) == 1:
            targets.setdefault(pk[0].lower(), []).append(t)
    for t, cs in schema.columns.items():
        for k, raw in cs.items():
            # An undeclared equality requires an unambiguous named single-column
            # primary key and a column identifying that entity in another table.
            owners = targets.get(k, [])
            if len(owners) != 1 or owners[0] == t or k in ("id", "name", "code", "key"):
                continue
            u = owners[0]
            if not (words(raw) & words(u)) or not re.search(r"(?:_?id|_code|_key)$", k):
                continue
            if any(n == u and a.lower() == k for n, a, b, _ in edges[t]):
                continue
            b = schema.pk[u][0]
            edges[t].append((u, raw, b, True))
            edges[u].append((t, b, raw, True))
    return edges


def route(edges, origins, target, q):
    queue = deque((t, {t}, []) for t in sorted(origins))
    candidates = []
    while queue:
        t, seen, path = queue.popleft()
        if t == target:
            # Question words distinguish multiple FKs between the same entities.
            role = sum(
                len(((words(a) | words(b)) - (words(s) | words(u))) & q)
                for s, a, u, b, _ in path
            )
            cost = (
                2 * len(path)
                + sum(int(implicit) for _, _, _, _, implicit in path)
                - min(2, role)
            )
            candidates.append((cost, tuple(path)))
            continue
        if len(path) >= 3:
            continue
        for u, a, b, implicit in edges[t]:
            if u not in seen | origins:
                queue.append((u, seen | {u}, path + [(t, a, u, b, implicit)]))
    ordered = sorted(set(candidates))
    if not ordered or (len(ordered) > 1 and ordered[0][0] == ordered[1][0]):
        return []
    return ordered[0][1]


def role_join(schema, cols, q):
    if len(cols) < 2 or any(f or c == "*" for f, c, _ in cols):
        return
    if re.search(
        r"""\d|['"]|\b(?:each|every|per|who (?:has|have)|where|having|most|least|highest|lowest|distinct|unique|more|less|only|either|both|not|never|except|or|between|after|before|starting|total|average)\b""",
        q,
        re.I,
    ):
        return
    query, *sort = re.split(
        r"\border(?:ed)?\s+(?:(?:in\s+)?(?:ascending|descending)\s+)?(?:by|according to)\b",
        q,
        maxsplit=1,
        flags=re.I,
    )
    if not sort and re.search(r"\b(?:order|ascending|descending|sort)\b", q, re.I):
        return
    rolewords = words(query)
    owners = []
    for _, name, _ in cols:
        options = []
        for t, cs in schema.columns.items():
            if name.lower() in cs:
                w = words(t)
                score = (
                    4 * len(w & rolewords)
                    + int(w <= rolewords)
                    + int(name.lower() in {k.lower() for k in schema.pk[t]})
                )
                options.append((score + 1, t))
        t = choose(options)
        if not t:
            return
        owners.append(t)
    if len(set(owners)) < 2 or len({c.lower() for _, c, _ in cols}) < len(cols):
        return
    edges = graph(schema)
    root = owners[0]
    joined = {root}
    bindings = []
    for target in dict.fromkeys(owners[1:]):
        if target not in joined:
            path = route(edges, joined, target, rolewords)
            if not path:
                return
            bindings.extend(path)
            joined.update(e[2] for e in path)
    role = any(
        ((words(a) | words(b)) - words(s) - words(t)) & rolewords
        for s, a, t, b, _ in bindings
    )
    if not (role or len(bindings) >= 2 or sort):
        return
    aliases = {root: "r0", **{e[2]: "r" + str(i + 1) for i, e in enumerate(bindings)}}
    projections = ", ".join(col(aliases[t], c) for (_, c, _), t in zip(cols, owners))
    sql = f"SELECT {projections} FROM {ident(root)} AS r0"
    for s, a, t, b, _ in bindings:
        sql += f" JOIN {ident(t)} AS {aliases[t]} ON {col(aliases[s], a)} = {col(aliases[t], b)}"
    if sort:
        opts = []
        for t in joined:
            for c in schema.columns[t].values():
                hits = words(c) & words(sort[0])
                if hits:
                    opts.append((len(hits) + len(hits) / len(words(c)), (t, c)))
        key = choose(opts)
        if not key:
            return
        t, c = key
        sql += (
            " ORDER BY "
            + col(aliases[t], c)
            + (" DESC" if re.search(r"\bdesc(?:ending)?\b", q, re.I) else " ASC")
        )
    yield Plan(
        "role_join",
        sql,
        tuple(sorted(joined)),
        tuple(dict.fromkeys(k for _, a, _, b, _ in bindings for k in (a, b))),
    )
