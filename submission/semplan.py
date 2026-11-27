"""Compile bounded relational plans from question spans, public headers, and schema."""

import re
from dataclasses import dataclass
from repair.edits import AGG, IDENT

STOP = set(
    "the a an to in of for on any all have has had do does did that who which whose are is with at by and than from many how number show list what name names id ids record records".split()
)
NEG = re.compile(
    r"\b(?:do(?:es)? not|did not|don['’]t|doesn['’]t|without|never|not (?:used|involved|in)|no)\b",
    re.I,
)
THRESH = re.compile(
    r"\b(at least|at most|more than|less than|fewer than)\s+(\d+|one|two|three|four|five)\b",
    re.I,
)
OPS = {
    "at least": ">=",
    "at most": "<=",
    "more than": ">",
    "less than": "<",
    "fewer than": "<",
}
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def stem(word):
    if word in ("people", "men", "women", "children"):
        return {
            "people": "person",
            "men": "man",
            "women": "woman",
            "children": "child",
        }[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "sses", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def words(s):
    terms = re.findall(r"[a-zA-Z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s))
    return {stem(t.lower()) for t in terms if t.lower() not in STOP}


def ident(s):
    return '"' + s.replace('"', '""') + '"'


def col(alias, name):
    return alias + "." + ident(name)


def literal(value):
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class Plan:
    family: str
    sql: str
    tables: tuple
    keys: tuple


class Schema:
    def __init__(self, records):
        self.columns = {
            r["name"]: {c["name"].lower(): c["name"] for c in r["columns"]}
            for r in records
        }
        self.types = {
            r["name"]: {c["name"]: c["type"].upper() for c in r["columns"]}
            for r in records
        }
        self.pk = {
            r["name"]: [c["name"] for c in r["columns"] if c["pk"]] for r in records
        }
        actual = {t.lower(): t for t in self.columns}
        self.edges = {t: [] for t in self.columns}
        for r in records:
            t = r["name"]
            for fk in r["foreign_keys"]:
                other = actual.get(fk["table"].lower())
                a, b = fk["from"], fk["to"]
                if (
                    other
                    and a
                    and b
                    and a.lower() in self.columns[t]
                    and b.lower() in self.columns[other]
                ):
                    self.edges[other].append((t, b, a, True))
                    self.edges[t].append((other, a, b, False))


def headers(names):
    out = []
    for h in names:
        m = AGG.fullmatch(h)
        if m:
            out.append((m[1].upper(), m[3], bool(m[2])))
        elif re.fullmatch(IDENT, h):
            out.append(("", h, False))
        else:
            return []
    return out


def project(cols, alias):
    out = []
    for fn, name, distinct in cols:
        arg = "*" if name == "*" else col(alias, name)
        out.append(f"{fn}({'DISTINCT ' if distinct else ''}{arg})" if fn else arg)
    return ", ".join(out)


def choose(scored):
    """Abstain if two different bindings tie for first place."""
    ordered = sorted(set(scored), reverse=True)
    if (
        not ordered
        or ordered[0][0] <= 0
        or (len(ordered) > 1 and ordered[0][0] == ordered[1][0])
    ):
        return None
    return ordered[0][1]


def roots(schema, cols, subject):
    names = {c.lower() for _, c, _ in cols if c != "*"}
    tokens = words(subject)
    options = []
    for t, available in schema.columns.items():
        if names <= available.keys():
            table = words(t)
            match = len(table & tokens)
            score = (
                4 * match
                + int(table <= tokens and bool(table))
                - 0.15 * len(table)
                + int(bool(names))
            )
            options.append((score, t))
    return choose(options)


def child(schema, root, phrase, subject):
    targets = words(phrase)
    origin = words(root) | words(subject)
    options = []
    for t, pk, fk, inverse in schema.edges[root]:
        if not inverse or t == root:
            continue
        relation = words(t)
        related = set().union(
            *(
                words(other)
                for other, _, _, rev in schema.edges[t]
                if not rev and other != root
            )
        )
        hits = (relation | related) & targets
        if not hits:
            continue
        # Prefer an association that names both participant roles to unrelated children.
        score = (
            4 * len(relation & targets)
            + 2 * len(related & targets)
            + len(relation & origin)
            + len(words(fk) & targets)
        )
        options.append((score, (t, pk, fk)))
    return choose(options)


def agentive_stem(word):
    # Nominal agent roles and verbs often share a stem after -er/-or.
    base = re.sub(r"(er|or)$", "", word).rstrip("e")
    return base if len(base) >= 4 else ""


def implicit_child(schema, root, tail, subject):
    if not words(root) & words(subject) or len(schema.pk[root]) != 1:
        return
    pk = schema.pk[root][0]
    if "INT" not in schema.types[root][pk]:
        return
    bindings = []
    for t, cs in schema.columns.items():
        if (
            root == t
            or not words(t) <= words(tail)
            or any(u == t for u, _, _, _ in schema.edges[root])
        ):
            continue
        verbs = {agentive_stem(w) for w in words(tail) - words(t)} - {""}
        for fk in cs.values():
            roles = {agentive_stem(w) for w in words(fk) if w.endswith(("er", "or"))}
            if (
                verbs & roles
                and "INT" in schema.types[t][fk]
                and fk not in schema.pk[t]
            ):
                bindings.append((t, pk, fk))
    return bindings[0] if len(bindings) == 1 else None


def anti(schema, cols, q, m):
    prefix, tail = q[: m.start()], q[m.end() :].strip(" ?.!")
    root = roots(schema, cols, prefix)
    if not root or re.search(
        r"""\b(?:in|with|having|over|under|than|located|called)\s+|["'\d]""", tail, re.I
    ):
        return
    bound = child(schema, root, tail, prefix) or implicit_child(
        schema, root, tail, prefix
    )
    if not bound:
        return
    t, pk, fk = bound
    competing = [b for other, a, b, _ in schema.edges[root] if other == t and a == pk]
    if len(set(competing)) > 1 and not (words(fk) & (words(tail) - words(t))):
        return
    sql = f"SELECT {project(cols, 'a')} FROM {ident(root)} AS a WHERE NOT EXISTS (SELECT 1 FROM {ident(t)} AS b WHERE {col('b', fk)} = {col('a', pk)})"
    yield Plan("anti", sql, (root, t), (pk, fk))


def frequency(schema, cols, q, m):
    prefix = q[: m.start()]
    tail = re.split(
        r"\b(?:in (?:ascending|descending|alphabetical)|order(?:ed)? by|sort(?:ed)? by|give me|list the)\b|[?.!,]",
        q[m.end() :],
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    if not re.fullmatch(r"[a-zA-Z -]{1,35}", tail) or re.search(
        r"\b(?:with|older|younger|of height|or|and|per|from)\b", tail, re.I
    ):
        return
    root = roots(schema, cols, prefix)
    if not root or not all(fn in ("", "COUNT") for fn, _, _ in cols):
        return
    fields = [n for f, n, _ in cols if not f]
    if not fields:
        return
    if re.search(
        r"\b(?:and (?:have|has|with)|who (?:use|used)|that (?:use|used))\b",
        prefix,
        re.I,
    ):
        return
    relation = None if words(tail) & words(root) else child(schema, root, tail, prefix)
    frm = ident(root) + " AS a"
    tables, keys = (root,), ()
    if relation:
        t, pk, fk = relation
        frm += f" JOIN {ident(t)} AS b ON {col('a', pk)} = {col('b', fk)}"
        tables, keys = (root, t), (pk, fk)
    elif not (
        words(tail) & words(root) or re.fullmatch(r"records?|times?", tail, re.I)
    ):
        return
    threshold = NUM.get(m[2].lower(), m[2])
    group = ", ".join(col("a", k) for k in fields)
    order = ""
    if re.search(r"\b(?:ascending|descending|alphabetical)\b", q, re.I):
        order = (
            " ORDER BY "
            + col("a", fields[0])
            + (" DESC" if "descending" in q.lower() else " ASC")
        )
    sql = f"SELECT {project(cols, 'a')} FROM {frm} GROUP BY {group} HAVING COUNT(*) {OPS[m[1].lower()]} {threshold}{order}"
    if relation and words(root) & words(prefix) and schema.pk[root]:
        entity = ", ".join(col("a", k) for k in schema.pk[root])
        # Group by entity identity even when two entities share a display name.
        # Count unique participants when the bridge declares their key.
        participants = [
            key
            for other, key, _, rev in schema.edges[t]
            if not rev and other != root and words(other) & words(tail)
        ]
        count = "COUNT(*)"
        if len(set(participants)) == 1:
            count = "COUNT(DISTINCT " + col("b", participants[0]) + ")"
        heading = project(cols, "a").replace("COUNT(*)", count)
        keyed = f"SELECT {heading} FROM {frm} GROUP BY {entity} HAVING {count} {OPS[m[1].lower()]} {threshold}{order}"
        if keyed != sql:
            yield Plan(
                "count_group",
                keyed,
                tables,
                tuple(dict.fromkeys((*keys, *schema.pk[root], *participants))),
            )
    yield Plan("count_group", sql, tables, keys)


def plans(q, public_columns, records):
    neg, threshold = NEG.search(q), THRESH.search(q)
    coord = re.search(r"\b(both|either)\b", q, re.I)
    if not (threshold or neg or coord or len(public_columns) >= 2):
        return []
    schema, cols = Schema(records), headers(public_columns)
    if not cols:
        return []
    if not (threshold or neg or coord):
        from paths import role_join

        return list(role_join(schema, cols, q))
    from scopes import detail_group, set_scopes

    scoped = list(set_scopes(schema, cols, q))
    if scoped:
        return scoped
    if coord:
        return []
    if neg:
        return list(anti(schema, cols, q, neg))[:2]
    detail = list(detail_group(schema, cols, q, threshold))
    return detail or list(frequency(schema, cols, q, threshold))[:2]
