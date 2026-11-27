"""Independent softmax models of SQL properties, conditioned on language and drafts."""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from sqlglot import exp
from repair.edits import core, main_select, parse, tables
from repair.sketch import sketch_tokens

IMPORTANCE = {
    "joins": 0.55,
    "distinct": 1.0,
    "group": 0.9,
    "group_width": 0.4,
    "order": 0.8,
    "limit": 0.35,
    "where": 0.8,
    "having": 0.6,
    "set": 1.0,
    "subquery": 0.65,
    "pred": 0.5,
    "count": 0.45,
    "left": 0.7,
}


def attributes(t):
    if t is None:
        return {}
    s = main_select(t)
    g = s.args.get("group")
    order = s.args.get("order")
    wh = s.args.get("where")
    h = s.args.get("having")
    projected = {
        c.name.lower()
        for p in s.expressions
        if isinstance(core(p), exp.Column)
        for c in p.find_all(exp.Column)
    }
    gp = [c.name.lower() for c in g.find_all(exp.Column)] if g else []
    count = list(t.find_all(exp.Count))
    return {
        "joins": str(min(3, len(s.args.get("joins", [])))),
        "distinct": str(int(bool(s.args.get("distinct")))),
        "group": "none" if not g else "display" if set(gp) <= projected else "key",
        "group_width": str(min(2, len(gp))),
        "order": order.expressions[0].this.key
        if order and order.expressions
        else "none",
        "limit": str(int(bool(s.args.get("limit")))),
        "where": "none" if not wh else wh.this.key,
        "having": h.this.key if h else "none",
        "set": t.key if isinstance(t, exp.SetOperation) else "none",
        "subquery": str(int(bool(t.find(exp.Subquery)))),
        "pred": str(min(3, len(list(wh.find_all(exp.Predicate))))) if wh else "0",
        "count": "distinct"
        if any(isinstance(c.this, exp.Distinct) for c in count)
        else "star"
        if any(isinstance(c.this, exp.Star) for c in count)
        else "column"
        if count
        else "none",
        "left": str(int(any(j.side == "LEFT" for j in s.args.get("joins", [])))),
    }


def context(obs, cat):
    q = obs["task"]["question"].lower()
    ex = obs["example"]
    text = sketch_tokens(q, ex["columns"], cat)
    text.update("q=" + w for w in re.findall(r"[a-z]+", q))
    text.update(
        (
            "bias",
            "ordered=" + str(obs["ordered"]),
            "empty=" + str(ex["row_count"] == 0),
            "draft_pass=" + str(obs["history"][0]["observation"]["matches_example"]),
        )
    )
    sources = [obs["issue_sql"]] + next(
        e["observation"] for e in obs["history"] if e["tool"] == "repair_hint"
    )
    props = [attributes(parse(s)) for s in sources]
    result = {}
    for head in IMPORTANCE:
        hits = Counter(p.get(head, "?") for p in props[1:])
        hv = hits.most_common(1)[0][0]
        dv = props[0].get(head, "?")
        result[head] = sorted(
            text
            | {
                "h=" + hv,
                "d=" + dv,
                "hd=" + hv + "/" + dv,
                "agree=" + str(hv == dv),
                "hint_consensus=" + str(len(hits) == 1),
            }
        )
    return result


@lru_cache(maxsize=1)
def parameters():
    return json.loads(Path(__file__).with_name("intent.json").read_text())


def logits(ctx, model):
    return {
        head: {
            label: sum(w.get(f, 0.0) for f in features)
            for label, w in model[head].items()
        }
        for head, features in ctx.items()
    }


def fidelity(tree, pred):
    attrs = attributes(tree)
    score = 0
    for head, weight in IMPORTANCE.items():
        v = pred[head]
        score += weight * max(
            -3.0, v.get(attrs[head], min(v.values()) - 1.0) - max(v.values())
        )
    return score
