"""Predict the public effect of a projection or DISTINCT edit on a previously checked relation."""

from collections import Counter
from functools import lru_cache
from sqlglot import exp
from repair.edits import parse, emit, core
from repair.search import canonical


@lru_cache(maxsize=4096)
def relation(sql):
    t = parse(canonical(sql))
    if (
        not isinstance(t, exp.Select)
        or t.find(exp.Subquery)
        or t.args.get("limit")
        or t.args.get("offset")
    ):
        return None
    if any(not isinstance(core(e), exp.Column) for e in t.expressions):
        return None
    if any(t.args.get(k) for k in ("group", "having", "with", "qualify")):
        return None
    proj = tuple(emit(core(c)) for c in t.expressions)
    b = t.copy()
    for k in ("distinct", "order"):
        b.set(k, None)
    b.set("expressions", [exp.Star()])
    return emit(b), proj, bool(t.args.get("distinct"))


def implication(candidate, obs, checks):
    """Only infer bag equality on completely observed rows from the same FROM and WHERE."""
    u = relation(candidate)
    if u is None or obs["ordered"]:
        return 0
    for e in reversed(checks):
        fb = e["observation"]
        if not fb["ok"] or fb["truncated"]:
            continue
        v = relation(e["sql"])
        if not v or u[0] != v[0] or not set(u[1]) <= set(v[1]) or (v[2] and not u[2]):
            continue
        cols = [v[1].index(c) for c in u[1]]
        if len(v[1]) != len(fb["columns"]):
            continue
        rows = [tuple(row[i] for i in cols) for row in fb["rows"]]
        if u[2]:
            rows = list(dict.fromkeys(rows))
        example = obs["example"]
        if len(rows) != example["row_count"]:
            return -1
        if not example["truncated"]:
            return 1 if Counter(rows) == Counter(map(tuple, example["rows"])) else -1
    return 0
