"""Represent scoped filters supported by the compiler; an opaque form yields no proof."""

from functools import lru_cache
from sqlglot import exp
from repair.edits import parse, tables


def branches(expr, aliases):
    # A disjunction of conjunctions. IN over literal constants is an OR. Limit
    # the product to keep unusually complex hints inexpensive.
    if isinstance(expr, exp.Paren):
        return branches(expr.this, aliases)
    if isinstance(expr, (exp.And, exp.Or)):
        left = branches(expr.this, aliases)
        right = branches(expr.expression, aliases)
        if left is None or right is None or len(left) * len(right) > 16:
            return None
        if isinstance(expr, exp.Or):
            return left | right
        return frozenset(a | b for a in left for b in right)
    lhs = expr.this
    if not isinstance(lhs, exp.Column):
        return None
    owner = (
        aliases.get(lhs.table.lower())
        if lhs.table
        else next(iter(aliases.values()))
        if len(aliases) == 1
        else None
    )
    if not owner:
        return None
    if isinstance(expr, exp.In):
        if not expr.expressions or any(
            not isinstance(x, exp.Literal) for x in expr.expressions
        ):
            return None
        op, values = "eq", expr.expressions
    elif isinstance(
        expr, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)
    ) and isinstance(expr.expression, (exp.Boolean, exp.Literal)):
        op, values = expr.key, [expr.expression]
    else:
        return None
    return frozenset(
        frozenset([(owner, lhs.name.lower(), op, str(v.this).casefold())])
        for v in values
    )


def arm(node):
    if (
        not isinstance(node, exp.Select)
        or node.args.get("group")
        or node.find(exp.Subquery)
    ):
        return None
    where = node.args.get("where")
    return branches(where.this, tables(node)) if where else frozenset([frozenset()])


@lru_cache(maxsize=1024)
def scope(sql):
    t = parse(sql)
    if isinstance(t, (exp.Intersect, exp.Except, exp.Union)):
        a, b = arm(t.this), arm(t.expression)
        if a is None or b is None:
            return None
        if isinstance(t, exp.Union):
            return "rows", a | b
        if isinstance(t, exp.Intersect):
            return "intersection", frozenset((a, b))
        return "group_exclusion", (a, b)
    a = arm(t)
    return ("rows", a) if a is not None else None


def comparison(plan, sql):
    """True for agreement, False for explicit conflict, None for unfamiliar forms."""
    if plan.family in ("intersection", "disjunction", "group_exclusion"):
        expected, actual = scope(plan.sql), scope(sql)
        if expected is None or actual is None:
            return None
        return expected == actual
    if plan.family == "role_join":
        a, b = parse(plan.sql), parse(sql)
        if not isinstance(b, exp.Select) or b.find(exp.Subquery) or b.args.get("group"):
            return None

        # A role-qualified join and an additional row filter are separate claims.
        # Require the same participants and bound join columns before comparing filters.
        def join_keys(t):
            names = tables(t)
            out = set()
            for j in t.args.get("joins") or []:
                e = j.args.get("on")
                if not isinstance(e, exp.EQ) or not all(
                    isinstance(v, exp.Column) and v.table.lower() in names
                    for v in (e.this, e.expression)
                ):
                    return None
                out.add(
                    frozenset(
                        (names[v.table.lower()], v.name.lower())
                        for v in (e.this, e.expression)
                    )
                )
            return out

        expected = join_keys(a)
        if (
            expected
            and expected == join_keys(b)
            and b.args.get("where")
            and not a.args.get("where")
        ):
            return False
    return None
