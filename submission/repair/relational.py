"""Schema-grounded repairs for column binding, join paths, filters and grouping."""

import re
from collections import defaultdict, deque

from sqlglot import exp

from repair.edits import header, projection, tables


def name_words(s):
    return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()))


def alignment(name, q):
    parts = name_words(name)
    return sum(
        bool(re.search(r"\b" + re.escape(w.rstrip("s")) + r"(?:s|es)?\b", q))
        for w in parts
        if len(w) > 1
    )


class Catalog:
    def __init__(self, records):
        self.columns = {}
        self.pk = {}
        self.graph = defaultdict(list)
        for t in records:
            table = t["name"].lower()
            self.columns[table] = {c["name"].lower() for c in t["columns"]}
            self.pk[table] = {c["name"].lower() for c in t["columns"] if c["pk"]}
            for fk in t["foreign_keys"]:
                if fk["to"] and fk["from"]:
                    other, a, b = (
                        fk["table"].lower(),
                        fk["from"].lower(),
                        fk["to"].lower(),
                    )
                    self.graph[table].append((other, a, b))
                    self.graph[other].append((table, b, a))

    def path(self, origins, target):
        queue = deque((t, []) for t in origins)
        visited = set(origins)
        while queue:
            t, path = queue.popleft()
            if t == target:
                return path
            for n, a, b in self.graph[t]:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [(t, a, n, b)]))
        return None


def grounded(tree, catalog):
    """Fix qualifiers within one SELECT scope and add short declared FK paths."""
    if tree.find(exp.Subquery) or tree.args.get("with"):
        return None
    out = tree.copy()
    names = tables(out)
    if not names or any(t not in catalog.columns for t in names.values()):
        return None
    cols = list(out.find_all(exp.Column))
    output_alias = {p.alias.lower() for p in out.expressions if p.alias}
    by_table = {table: alias for alias, table in names.items()}
    changed = False
    for c in cols:
        col, alias = c.name.lower(), c.table.lower()
        if (not alias and col in output_alias) or col == "*":
            continue
        if alias and col in catalog.columns.get(names.get(alias, ""), set()):
            continue
        owners = [a for a, t in names.items() if col in catalog.columns[t]]
        if owners:
            if alias not in owners and (alias or len(owners) > 1):
                c.set("table", exp.to_identifier(owners[0], quoted=True))
                changed = True
            continue
        options = []
        for t, columns in catalog.columns.items():
            if col in columns:
                path = catalog.path(by_table, t)
                if path and len(path) <= 2:
                    options.append((len(path), t, path))
        if not options:
            return None
        _, target, path = min(options)
        for a, x, b, y in path:
            condition = exp.column(x, table=by_table[a], quoted=True).eq(
                exp.column(y, table=b, quoted=True)
            )
            out = out.join(exp.to_table(b, quoted=True), on=condition, copy=False)
            by_table[b] = b
            names[b] = b
        c.set("table", exp.to_identifier(target, quoted=True))
        changed = True
    return out if changed else None


def prune(tree, catalog, question):
    """Remove peripheral join tables that supply no expression outside ON."""
    if not tree.args.get("joins") or tree.find(exp.Subquery) or tree.args.get("with"):
        return None
    names = tables(tree)
    if len(names) != 1 + len(tree.args["joins"]):
        return None
    if any(isinstance(p, exp.Star) for p in tree.expressions):
        return None
    roots = [tree.args["from"].this] + [j.this for j in tree.args["joins"]]
    required = set()
    for clause in ("expressions", "where", "group", "order", "having"):
        items = tree.args.get(clause)
        for item in items if isinstance(items, list) else [items]:
            if not item:
                continue
            for c in item.find_all(exp.Column):
                if c.table:
                    required.add(c.table.lower())
                else:
                    owners = [
                        a
                        for a, t in names.items()
                        if c.name.lower() in catalog.columns.get(t, set())
                    ]
                    if owners:
                        required.add(owners[0])
    if not required:
        return None
    # Keep the shortest connecting subtree among the sources serving the clauses.
    links = defaultdict(set)
    for j in tree.args["joins"]:
        if not j.args.get("on"):
            return None
        refs = {c.table.lower() for c in j.args["on"].find_all(exp.Column) if c.table}
        if len(refs) != 2:
            return None
        a, b = sorted(refs)
        links[a].add(b)
        links[b].add(a)
    kept = set(names)
    for _ in names:
        leaves = {a for a in kept - required if len(links[a] & kept) <= 1}
        if not leaves:
            break
        kept -= leaves
    if not kept or kept == set(names):
        return None
    first = next(r for r in roots if r.alias_or_name.lower() in kept)
    out = tree.copy()
    out.set("from", exp.From(this=first.copy()))
    out.set(
        "joins",
        [
            j.copy()
            for j in tree.args["joins"]
            if j.this.alias_or_name.lower() in kept
            and j.this.alias_or_name.lower() != first.alias_or_name.lower()
        ],
    )
    return out


def relational_variants(tree, q, example, cat):
    if not isinstance(tree, exp.Select):
        return
    fixed = grounded(tree, cat)
    if fixed is not None:
        fixed = projection(fixed, example["columns"], cat.columns) or fixed
        yield fixed, "bind"
    base = projection(tree, example["columns"], cat.columns) or tree
    pruned = prune(base, cat, q)
    if pruned is not None:
        yield pruned, "prune"
    if base.args.get("group"):
        # Extremum over grouped entities: group by the key on either side of an equality.
        for j in base.args.get("joins", []):
            on = j.args.get("on")
            if isinstance(on, exp.EQ):
                for col in (on.this, on.expression):
                    if isinstance(col, exp.Column):
                        out = base.copy()
                        out.set("group", exp.Group(expressions=[col.copy()]))
                        yield out, "group_key"
    group_cue = re.search(
        r"\b(most common|least common|most frequent\w*|least frequent\w*|at least|at most|more than|fewer than|less than|each|per)\b",
        q,
    )
    if group_cue:
        # A one-table frequency query with a projected category as grouping key.
        target = [header(h) for h in example["columns"]]
        if all(isinstance(x, (exp.Column, exp.Count)) for x in target):
            cols = [x for x in target if isinstance(x, exp.Column)]
            for t in tables(tree).values():
                if not cols or not all(
                    x.name.lower() in cat.columns.get(t, set()) for x in cols
                ):
                    continue
                out = exp.select(*(x.copy() for x in target)).from_(
                    exp.to_table(t, quoted=True)
                )
                out = out.group_by(*(x.copy() for x in cols))
                count = exp.Count(this=exp.Star())
                m = re.search(
                    r"\b(at least|at most|more than|fewer than|less than) (\d+|one|two|three|four|five)\b",
                    q,
                )
                if m:
                    value = dict(one=1, two=2, three=3, four=4, five=5).get(m[2], m[2])
                    op = {
                        "at least": exp.GTE,
                        "at most": exp.LTE,
                        "more than": exp.GT,
                        "fewer than": exp.LT,
                        "less than": exp.LT,
                    }[m[1]]
                    out = out.having(
                        op(this=count, expression=exp.Literal.number(value))
                    )
                elif re.search(r"\b(?:most|least) (?:common|frequent)", q):
                    out = out.order_by(
                        exp.Ordered(this=count, desc="least" not in q)
                    ).limit(1)
                elif not any(isinstance(x, exp.Count) for x in target):
                    continue
                yield out, "frequency"
    where = base.args.get("where")
    if where and re.search(r"\b(both|but not|never|either)\b", q):
        cond = where.this.unnest()
        if isinstance(cond, (exp.And, exp.Or)):
            left = base.copy().where(cond.this.copy(), append=False)
            right = base.copy().where(cond.expression.copy(), append=False)
            if "but not" in q:
                yield left.except_(right), "set"
            elif "both" in q:
                yield left.intersect(right), "set"
            elif "either" in q:
                yield left.union(right), "set"
