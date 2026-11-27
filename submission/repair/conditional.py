"""Rank an untried SQL given discrepancies measured by a previous public check."""

import json
from functools import lru_cache
from pathlib import Path
from repair.preferences import properties
from repair.edits import parse


@lru_cache(maxsize=1)
def parameters():
    return json.loads(Path(__file__).with_name("conditional.json").read_text())


@lru_cache(maxsize=2048)
def signature(sql):
    a, r = properties(sql)
    tree = parse(sql)
    # Anonymize edit features only after comparing the two queries' actual identifiers.
    r = r.copy()
    if tree is not None:
        from sqlglot import exp

        r["predicate"] = frozenset(
            (
                p.key,
                p.this.sql().lower() if p.this else "",
                p.expression.sql().lower() if p.expression else "",
            )
            for p in tree.find_all(exp.Predicate)
            if isinstance(p.this, exp.Expression)
        )
    return a, r


def discrepancy(o, fb):
    if not fb["ok"]:
        e = (fb.get("error") or {}).get("message", "").lower()
        return (
            "binding"
            if any(s in e for s in ("no such", "ambiguous", "column"))
            else "error"
        )
    target = o["example"]
    if fb["matches_example"]:
        return "weakpass" if target["rows"] in ([], [[0]], [[None]]) else "pass"
    if len(fb["columns"]) != len(target["columns"]):
        return "width"
    if fb["row_count"] == 0:
        return "empty"
    if fb["row_count"] < target["row_count"]:
        return "few"
    if fb["row_count"] > target["row_count"]:
        return "extra"
    return "values"


def relative(sql, origin, kind):
    a, r = signature(sql)
    b, t = signature(origin)
    x = {}
    for role in sorted(r.keys() | t.keys()):
        u = r.get(role, set())
        v = t.get(role, set())
        change = (
            "same" if u == v else "add" if v < u else "remove" if u < v else "replace"
        )
        x[kind + "/" + role + "/" + change] = 1
    for k in a:
        before = b.get(k, "?")
        if before != a[k]:
            x[kind + "/change/" + k] = 1
            x[kind + "/" + k + "/" + before + ">" + a[k]] = 1
    return x


def inputs(sql, base, o, anchor):
    kind = discrepancy(o, anchor["observation"])
    x = relative(sql, anchor["sql"], kind)
    for k in (
        "votes",
        "draft",
        "joins",
        "invalid",
        "declared_fk",
        "extra_leaf",
        "match_literal",
        "distance",
        "prior",
        "missing_entities",
        "bad_link",
        "group_pk",
        "gen_filter_graph",
        "gen_plain",
        "gen_legacy",
        "gen_template",
        "extra_literal",
        "distinct",
    ):
        if base.get(k):
            x[kind + "/base/" + k] = base[k]
    return x


def estimate(sql, x, o, checks):
    from repair.preferences import score

    # Each distinct discrepancy supplies one observed context; use the latest of that kind.
    anchors = {discrepancy(o, e["observation"]): e for e in checks}
    w = parameters()
    return sum(score(inputs(sql, x, o, e), w) for e in anchors.values()) / len(anchors)
