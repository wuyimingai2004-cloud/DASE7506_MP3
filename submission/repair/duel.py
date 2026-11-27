"""Pair-dependent comparison for SQLs that pass the same public example."""

import json
from functools import lru_cache
from pathlib import Path
from repair.linker import evidence, role_sets
from repair.preferences import properties


@lru_cache(maxsize=1)
def parameters():
    return json.loads(Path(__file__).with_name("duel.json").read_text())


def candidate(sql, base, prob):
    attrs, _ = properties(sql)
    roles = role_sets(sql)
    dense = {k: v for k, v in base.items() if "/" not in k}
    dense.update(evidence(roles, prob))
    return attrs, roles, dense


def difference(a, b):
    aa, ar, ax = a
    ba, br, bx = b
    differing = [r for r in ar if ar[r] != br[r]]
    ac = [r for r in aa if aa[r] != ba[r]]
    gates = []
    if len(differing) <= 1:
        gates.append("roles:" + "".join(differing))
    if len(ac) <= 1:
        gates.append("attr:" + "".join(ac))
    if ar["table"] == br["table"]:
        gates.append("same_tables")
    if ar["output"] == br["output"]:
        gates.append("same_output")
    if aa["group"] != ba["group"]:
        gates.append("group_change")
    if aa["distinct"] != ba["distinct"]:
        gates.append("distinct_change")
    out = {}
    for k in sorted(ax.keys() | bx.keys()):
        d = max(-4, min(4, ax.get(k, 0.0) - bx.get(k, 0.0)))
        if not d:
            continue
        out[k] = d
        if k.startswith("link/") or k in (
            "votes",
            "draft",
            "extra_leaf",
            "declared_fk",
            "bad_link",
            "group_pk",
            "distinct",
            "prior",
            "distance",
            "tool_exact_headers",
            "missing_entities",
        ):
            for g in gates:
                out[g + "|" + k] = d
    return out
