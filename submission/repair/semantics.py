"""Additional semantic hypotheses using the question, declared keys and SQL expressions."""

import re
from sqlglot import exp
from repair.edits import core, header, projection, tables
from repair.relational import prune


def semantic_variants(tree, q, example, cat):
    if not isinstance(tree, exp.Select):
        return
    base = projection(tree, example["columns"], cat.columns) or tree
    # Conventional inner joins and bare COUNT(*) are frequent in the training references.
    if any(j.side in ("LEFT", "RIGHT") for j in base.args.get("joins", [])):
        out = base.copy()
        for j in out.args.get("joins", []):
            j.set("side", None)
        yield out, "inner"
    if base.args.get("group") or base.args.get("having"):
        out = base.copy()
        for count in list(out.find_all(exp.Count)):
            count.set("this", exp.Star())
        yield out, "count"
    if base.args.get("order") and base.args.get("limit") and not base.args.get("group"):
        by = base.args["order"].expressions[0]
        if isinstance(by.this, exp.Column) and "most" in q:
            group = [
                core(p).copy()
                for p in base.expressions
                if isinstance(core(p), exp.Column)
            ]
            if group:
                out = base.copy()
                out.set("group", exp.Group(expressions=group))
                out.args["order"].expressions[0].set(
                    "this", exp.Sum(this=by.this.copy())
                )
                yield out, "group_sum"
    for node in base.find_all(exp.In):
        if (
            len(node.expressions) == 2
            and "both" in q
            and all(isinstance(e, exp.Literal) for e in node.expressions)
        ):
            if base.args.get("where") and base.args["where"].this is node:
                left = base.copy().where(
                    node.this.copy().eq(node.expressions[0].copy()), append=False
                )
                right = base.copy().where(
                    node.this.copy().eq(node.expressions[1].copy()), append=False
                )
                yield left.intersect(right), "set"
    negative = bool(
        re.search(r"\b(not|never|no|without|except|don't|doesn't|haven't|didn't)\b", q)
    )
    if negative:
        for node in base.find_all(exp.Not):
            if isinstance(node.this, exp.In) and node.this.args.get("query"):
                sub = node.this.args["query"].this
                if (
                    isinstance(sub, exp.Select)
                    and sub.args.get("where")
                    and sub.args["where"].find(exp.Null)
                ):
                    out = base.copy()
                    for sub in out.find_all(exp.Select):
                        if (
                            sub is not out
                            and sub.args.get("where")
                            and sub.args["where"].find(exp.Null)
                        ):
                            sub.set("where", None)
                    yield out, "anti"
                    break
        where = base.args.get("where")
        if where and isinstance(where.this, (exp.NEQ, exp.EQ)):
            pred = where.this
            if isinstance(pred.expression, exp.Literal):
                right = base.copy().where(
                    pred.this.copy().eq(pred.expression.copy()), append=False
                )
                left = base.copy()
                left.set("where", None)
                left = prune(left, cat, q) or left
                yield left.except_(right), "anti"
    # Quantified comparisons: reduce a scalar subquery to the relevant extremum.
    if re.search(r"\b(any|all|every)\b", q):
        for op in (exp.GT, exp.LT, exp.GTE, exp.LTE):
            for node in base.find_all(op):
                if isinstance(node.expression, exp.Subquery) and isinstance(
                    node.expression.this, exp.Select
                ):
                    s = node.expression.this
                    if len(s.expressions) != 1 or not isinstance(
                        core(s.expressions[0]), exp.Column
                    ):
                        continue
                    greater = isinstance(node, (exp.GT, exp.GTE))
                    for agg in [exp.Max, exp.Min] if greater else [exp.Min, exp.Max]:
                        out = base.copy()
                        for p in out.find_all(op):
                            if p.sql() == node.sql():
                                sub = p.expression.this
                                sub.set(
                                    "expressions",
                                    [agg(this=core(s.expressions[0]).copy())],
                                )
                                break
                        yield out, "extreme"
    # If only the result category varies, grouping that category counts its occurrences.
    if (
        len(example["columns"]) == 1
        and not base.args.get("group")
        and re.search(r"\b(most|least|fewest)\b", q)
    ):
        col = header(example["columns"][0])
        if isinstance(col, exp.Column):
            for t in tables(tree).values():
                if col.name.lower() in cat.columns.get(t, set()):
                    out = (
                        exp.select(col.copy())
                        .from_(exp.to_table(t, quoted=True))
                        .group_by(col.copy())
                    )
                    desc = "least" not in q and "fewest" not in q
                    yield (
                        out.order_by(
                            exp.Ordered(this=exp.Count(this=exp.Star()), desc=desc)
                        ).limit(1),
                        "frequency",
                    )
    # A hint may introduce a join solely to compare an entity name. Keep its literal
    # but bind it to a name column of another already-mentioned entity.
    where = base.args.get("where")
    if where and not base.find(exp.Subquery):
        for idx, node in enumerate(base.find_all(exp.EQ)):
            if not (
                isinstance(node.expression, exp.Literal)
                and node.expression.is_string
                and isinstance(node.this, exp.Column)
            ):
                continue
            for alias, t in tables(base).items():
                for col in sorted(cat.columns.get(t, ())):
                    if col.endswith("name") and (
                        col == "name" or col.replace("_", "").startswith(t[:3])
                    ):
                        out = base.copy()
                        p = list(out.find_all(exp.EQ))[idx]
                        p.set("this", exp.column(col, table=alias, quoted=True))
                        compact = prune(out, cat, q)
                        if compact is not None:
                            yield compact, "name_filter"
