"""Instantiate alternative column owners, foreign-key edges and correlated comparison edits."""

from sqlglot import exp
from repair.edits import core, emit, header, parse, projection, tables
from repair.sketch import from_graph
from transforms import eligible


def variants(tree, obs, cat):
    if not isinstance(tree, exp.Select):
        return
    for cmp in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE):
        quantifier = cmp.expression
        if isinstance(quantifier, (exp.All, exp.Any)):
            sub = quantifier.this.unnest()
            if isinstance(sub, exp.Select) and len(sub.expressions) == 1:
                for agg in (exp.Min, exp.Max):
                    target = sub.copy()
                    target.set(
                        "expressions", [agg(this=core(sub.expressions[0]).copy())]
                    )
                    yield (
                        tree.transform(
                            lambda n: target.subquery() if n == quantifier else n
                        ),
                        "quantified",
                        0.6,
                    )
    if not eligible(tree):
        return
    base = projection(tree, obs["example"]["columns"], cat.columns) or tree
    names = tables(base)
    if len(names) != len(set(names.values())) or not names:
        return
    # Multiple foreign keys between the same pair frequently encode different roles.
    for j, join in enumerate(base.args.get("joins", ())):
        eq = join.args.get("on")
        if not isinstance(eq, exp.EQ) or not all(
            isinstance(c, exp.Column) for c in (eq.this, eq.expression)
        ):
            continue
        a, b = eq.this.table.lower(), eq.expression.table.lower()
        for other, x, y in cat.graph.get(names.get(a), ()):
            if other == names.get(b) and (x, y) != (
                eq.this.name.lower(),
                eq.expression.name.lower(),
            ):
                v = base.copy()
                v.args["joins"][j].set(
                    "on",
                    exp.column(x, table=a, quoted=True).eq(
                        exp.column(y, table=b, quoted=True)
                    ),
                )
                yield v, "edge", 0.8
    # Same-named attributes on connected tables: change one projected role independently.
    for i, expr in enumerate(base.expressions):
        col = core(expr)
        if not isinstance(col, exp.Column):
            continue
        n = col.name.lower()
        for alias, t in names.items():
            if n in cat.columns.get(t, ()) and alias != col.table.lower():
                v = base.copy()
                core(v.expressions[i]).set(
                    "table", exp.to_identifier(alias, quoted=True)
                )
                yield v, "owner", 0.4
    # Reconstruct the required graph from every outside-ON attribute in the source.
    # Use observed schema FK paths even if the source has an invalid/disconnected ON.
    v = base.copy()
    v.set("from", None)
    v.set("joins", None)
    required = []
    for c in v.find_all(exp.Column):
        n = c.name.lower()
        owners = [t for t in names.values() if n in cat.columns.get(t, ())]
        t = names.get(c.table.lower())
        if t not in owners:
            if len(owners) != 1:
                return
            t = owners[0]
        c.set("table", exp.to_identifier(t, quoted=True))
        required.append(t)
    if not required:
        return
    required = list(dict.fromkeys(required))
    if len(required) > 1:
        built, _ = from_graph(v, required, cat)
        if built is not None:
            yield built, "path", 0.6
    # Explore one missing participation relation, including associative tables without
    # a column in the projection. Preserve WHERE/GROUP/HAVING/ORDER on the new graph.
    if base.args.get("group") or any(e.find(exp.Count) for e in base.expressions):
        for extra in sorted(
            {b for a in required for b, _, _ in cat.graph.get(a, ())} - set(required)
        ):
            built, _ = from_graph(v, required + [extra], cat)
            if built is not None:
                yield built, "participation", 0.2
