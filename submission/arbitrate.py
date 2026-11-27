"""Transfer the direction of ambiguous public-pass edits across unrelated TRAIN databases."""

from collections import Counter
from functools import lru_cache
import json
import math
import re
from pathlib import Path
from sqlglot import exp
from repair.edits import core, emit, parse, main_select, tables, projection_distance
from repair.search import canonical
from repair.preferences import properties
from belief import condition, descriptor


@lru_cache(maxsize=4096)
def vector(sql):
    t = parse(canonical(sql))
    if t is None:
        return ()
    s = main_select(t)
    return (
        len(tables(s)),
        int(bool(s.args.get("distinct"))),
        len(list(t.find_all(exp.Select))),
        len(list(t.find_all(exp.Predicate))),
        len(list(t.find_all(exp.Group))),
        len(list(t.find_all(exp.Having))),
        len(list(t.find_all(exp.Order))),
        len(list(t.find_all(exp.Limit))),
        len(list(t.find_all(exp.AggFunc))),
        len(list(t.find_all(exp.Literal))),
        len(list(t.find_all(exp.Lower))),
        sum(bool(j.side) for j in t.find_all(exp.Join)),
        len(s.expressions),
    )


def sign(n):
    return int(n > 0) - int(n < 0)


def contrast(a, b):
    va = vector(a)
    vb = vector(b)
    if not va or not vb:
        return "parse"
    delta = ",".join(
        str(i) + ("+" if x > y else "-")
        for i, (x, y) in enumerate(zip(va, vb))
        if x != y
    )
    return delta or "same_size"


def evidence(sql, o, fb, counts):
    k = canonical(sql)
    a = properties(sql)[1]
    names = a.get("table", ())
    question = o["task"]["question"].casefold()
    ref = o["example"]["columns"]
    return (
        int(ref == fb["columns"]),
        counts[k],
        -projection_distance(parse(sql), ref),
        sum(n.rstrip("s") in question for n in names),
        sum(
            bool(n) and n in question
            for n in a.get("literal", ())
            if isinstance(n, str)
        ),
        int(k == canonical(o["issue_sql"])),
    )


def keys(a, b, va, vb, o):
    op = contrast(a, b)
    if op == "parse":
        return []
    da = descriptor(a)
    db = descriptor(b)
    mask = sum(1 << i for i, (x, y) in enumerate(zip(da, db)) if x != y)
    group = condition(o)
    q = o["task"]["question"].lower()
    cue = "".join(
        str(int(bool(re.search(p, q))))
        for p in (
            r"\b(unique|different|distinct)\b",
            r"\b(each|per|every)\b",
            r"\b(no|not|never|without)\b",
        )
    )
    extra = ",".join(str(sign(x - y)) for x, y in zip(va, vb))
    return [
        f"e/{extra}",
        f"op/{op}",
        f"op/{op}/{mask}",
        f"op/{op}/{group}/{cue}",
        f"both/{op}/{extra}",
    ]


@lru_cache(maxsize=1)
def model():
    return json.loads(Path(__file__).with_name("arbitration.json").read_text())


def preference(a, b, va, vb, o):
    def estimate(k):
        yes, total = model().get(k, (0, 0))
        return (2 * yes - total) / (total + 5)

    ab = keys(a, b, va, vb, o)
    ba = keys(b, a, vb, va, o)
    weights = (0.8, 1, 1, 1.2, 1.3)
    return sum(w * (estimate(k) - estimate(l)) for w, k, l in zip(weights, ab, ba)) / (
        2 * sum(weights)
    )
