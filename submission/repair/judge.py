"""Two TRAIN-fitted heads for search and selection after real tool feedback."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from sqlglot import exp
from repair.edits import core, main_select, parse, signature, tables
from repair.search import dot
from repair.sketch import words


@lru_cache(maxsize=1)
def weights():
    return json.loads(Path(__file__).with_name("judge.json").read_text())


def features(sql, x, obs):
    t = parse(sql)
    sel = main_select(t)
    q = obs["task"]["question"].lower()
    n = obs["example"]["row_count"]
    src = tables(sel)
    ordered = bool(sel.args.get("order"))
    limit = bool(sel.args.get("limit"))
    negative = bool(re.search(r"\b(no|not|never|except|without|neither)\b", q))
    grouped = bool(sel.args.get("group"))
    sort = sel.args.get("order")
    having = sel.args.get("having")
    extreme = bool(
        re.search(
            r"\b(most|least|largest|smallest|highest|lowest|top|fewest|first|last|youngest|oldest|maximum|minimum|longest|shortest)\b",
            q,
        )
    )
    cnt = bool(re.search(r"\b(count|number|how many|times|frequen\w*|common)\b", q))
    each = bool(re.search(r"\b(each|per|every|for all)\b", q))
    qty = bool(
        re.search(
            r"\b(at least|less than|more than|at most|greater than|fewer than)\b", q
        )
    )
    nofilter = not sel.args.get("where")
    nonempty = n > 0 and obs["example"]["rows"] not in ([[0]], [[None]])
    lits = [v.this for v in t.find_all(exp.Literal) if v.is_string]
    draft = obs["history"][0]["observation"]
    header_ok = [re.sub(r"\s+", "", h).casefold() for h in obs["example"]["columns"]]
    star = any(isinstance(core(e), exp.Star) for e in sel.expressions)
    cond = sel.args.get("where")
    condwords = (
        words(" ".join(c.name for c in cond.find_all(exp.Column))) if cond else set()
    )
    x = x.copy()
    x.update(
        {
            "order_mismatch": int(ordered != obs["ordered"]),
            "ordered_cap": int(obs["ordered"] and limit),
            "agrees_extreme": int(extreme and limit),
            "unasked_limit": int(limit and not extreme),
            "ordered_count": int(bool(sort and sort.find(exp.Count)) and cnt),
            "group_count": int(grouped and (cnt or each)),
            "group_without_cue": int(grouped and not (cnt or each or qty or extreme)),
            "having_quantity": int(bool(having) and qty),
            "extra_having": int(bool(having) and not qty),
            "anti": int(
                negative and (isinstance(t, exp.Except) or bool(t.find(exp.Not)))
            ),
            "intersect_both": int(isinstance(t, exp.Intersect) and "both" in q),
            "distinct_question": int(
                bool(sel.args.get("distinct"))
                and not bool(re.search(r"\b(distinct|different|unique)\b", q))
            ),
            "where_grounded": min(5, len(condwords & words(q))),
            "pred_outside": min(5, len(condwords - words(q))),
            "extra_tbl_empty": int(not nonempty) * x["unused_unmentioned"],
            "has_wildcard": int(star),
            "many_unfiltered": int(n > 4 and nofilter),
            "draft_prior_ok": int(bool(x["draft"] and draft["matches_example"])),
            "hint_prior_ok": x["votes"] * int(not draft["matches_example"]),
            "exact_literal": sum(l in obs["task"]["question"] for l in lits),
            "lower_literal": sum(v.islower() for v in lits),
            "upper_literal": sum(v.isupper() for v in lits),
            "literal_wrapped": sum("%" in v for v in lits),
            "output_alias": sum(bool(p.alias) for p in sel.expressions),
            "schema_hint": int(len(src) > 1 and bool(x["votes"])),
            "nullfilter": int(bool(t.find(exp.Null))),
            "selects": min(4, len(list(t.find_all(exp.Select)))),
            "expected_agg": sum(
                int(bool(p.find(exp.AggFunc)))
                != int(bool(re.match(r"(count|avg|sum|max|min)\(", h)))
                for p, h in zip(sel.expressions, header_ok)
            ),
        }
    )
    return x


def feedback_features(obs, fb):
    headers = obs["example"]["columns"]
    cols = fb["columns"]
    match = [h == c for h, c in zip(headers, cols)]
    simple = lambda s: re.sub(r'[\s"`\[\]]+', "", s).lower()
    return {
        "tool_exact_headers": int(headers == cols),
        "tool_header_fraction": sum(match) / max(1, len(headers)),
        "tool_normal_headers": int(
            list(map(simple, headers)) == list(map(simple, cols))
        ),
        "tool_header_sparse": int(headers == cols and obs["example"]["row_count"] == 0),
    }


def score(x, head):
    w = weights()[head]
    return dot(x, w)


def evidence(sql, obs, x):
    """High-confidence structural evidence visible before issuing a tool call."""
    tree = parse(sql)
    sel = main_select(tree)
    src = tables(sel)
    records = next(e["observation"] for e in obs["history"] if e["tool"] == "schema")
    entities = {r["name"].lower(): r for r in records}
    fs = set()
    for t, r in entities.items():
        for e in r["foreign_keys"]:
            if e["from"] and e["to"]:
                a = (t, e["from"].lower())
                b = (e["table"].lower(), e["to"].lower())
                fs.update(((a, b), (b, a)))
    fk = bad = taut = 0
    for join in sel.args.get("joins", []):
        on = join.args.get("on")
        if (
            not isinstance(on, exp.EQ)
            or not isinstance(on.this, exp.Column)
            or not isinstance(on.expression, exp.Column)
        ):
            continue
        a, b = [
            (src.get(c.table.lower()), c.name.lower()) for c in (on.this, on.expression)
        ]
        if a == b and a[0]:
            taut += 1
        elif (a, b) in fs:
            fk += 1
        elif a[0] and b[0] and a[1] != b[1]:
            bad += 1
    q = obs["task"]["question"].lower()
    qs = words(q)
    active = {t: set() for t in src.values()}
    for c in sel.find_all(exp.Column):
        if c.find_ancestor(exp.Join) is None:
            if c.table.lower() in src:
                active[src[c.table.lower()]].add(c.name.lower())
            elif not c.table:
                for t in src.values():
                    if any(
                        c.name.lower() == col["name"].lower()
                        for col in entities.get(t, {}).get("columns", ())
                    ):
                        active[t].add(c.name.lower())
    extra = 0
    for t, cols in active.items():
        if cols:
            continue
        elsewhere = {w for other in active if other != t for w in words(other)}
        if not (words(t) - elsewhere) & qs:
            extra += 1
    leftnull = bool(
        re.search(r"\b(no|not|without|never)\b", q)
        and sel.args.get("where")
        and sel.args["where"].find(exp.Null)
        and any(j.side == "LEFT" for j in sel.args.get("joins", []))
    )
    sparse = obs["example"]["row_count"] == 0 or obs["example"]["rows"] in (
        [[0]],
        [[None]],
    )
    projection = sel.expressions
    dup_proj = len(projection) - len({p.sql().lower() for p in projection})
    # The ordered flag and projection labels are part of the official observation.
    # Empty public results give little evidence about these discrepancies.
    prior = (
        -3.0 * x["order_mismatch"]
        - 3.2 * taut
        - 1.15 * bad
        - 1.0 * extra
        - 0.75 * x["extra_num"]
        - 1.6 * x["invalid"]
        - 0.75 * dup_proj
        + 0.35 * leftnull
        + 0.4 * x["intersect_both"]
        + 0.55 * int(sparse) * x["case"]
        + 0.18 * int(fk > 0)
    )
    return {
        "declared_fk": min(4, fk),
        "bad_link": min(4, bad),
        "taut_join": min(4, taut),
        "extra_leaf": extra,
        "leftnull": int(leftnull),
        "duplicate_output": dup_proj,
    }, prior


def input_features(sql, x, obs):
    x = features(sql, x, obs)
    more, prior = evidence(sql, obs, x)
    return x | more, prior
