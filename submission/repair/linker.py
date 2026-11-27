"""Predict which anonymous schema slots participate in a reference query, from deployable observations."""

from functools import lru_cache
from math import log1p
from sqlglot import exp
from repair.edits import parse, tables
from repair.preferences import stem, IGNORE

ROLES = ("table", "output", "where", "group", "order", "having")


def schema_of(o):
    return next(e["observation"] for e in o["history"] if e["tool"] == "schema")


def role_sets(sql):
    tree = parse(sql)
    roles = {k: set() for k in ROLES}
    if tree is None:
        return roles
    roles["table"] = {n.name.lower() for n in tree.find_all(exp.Table)}
    for sel in tree.find_all(exp.Select):
        for role in ROLES[1:]:
            exprs = sel.expressions if role == "output" else [sel.args.get(role)]
            for ex in exprs:
                if ex is None:
                    continue
                for c in ex.find_all(exp.Column):
                    if c.find_ancestor(exp.Select) is sel:
                        roles[role].add(c.name.lower())
    return roles


def slot_inputs(o):
    import re

    q = o["task"]["question"]
    qw = stem(q)
    first = stem(
        re.split(
            r"\b(?:where|whose|who|with|that|having|which are|ordered by)\b",
            q,
            maxsplit=1,
            flags=re.I,
        )[0]
    )
    hdr = stem(" ".join(o["example"]["columns"]))
    schema = schema_of(o)
    owners = {}
    pk = set()
    fk = set()
    for t in schema:
        name = t["name"].lower()
        for c in t["columns"]:
            k = c["name"].lower()
            owners.setdefault(k, set()).add(name)
            if c["pk"]:
                pk.add(k)
        fk.update(f["from"].lower() for f in t["foreign_keys"] if f["from"])
    hints = next(e["observation"] for e in o["history"] if e["tool"] == "repair_hint")
    src = [role_sets(o["issue_sql"])] + [role_sets(h) for h in hints]
    out = {}
    for role in ROLES:
        domain = (
            sorted(t["name"].lower() for t in schema)
            if role == "table"
            else sorted(owners)
        )
        out[role] = {}
        for n in domain:
            w = stem(n)
            info = w - IGNORE
            hs = sum(n in z[role] for z in src[1:]) / len(hints)
            draft = int(n in src[0][role])
            val = {
                "bias": 1,
                "draft": draft,
                "support": hs,
                "unanimous": int(hs == 1),
                "split": int(0 < hs < 1),
                "literal_name": int(n.replace("_", " ") in q.lower()),
                "all_words": int(bool(w) and w <= qw),
                "coverage": len(w & qw) / max(1, len(w)),
                "focus": len(info & first) / max(1, len(info)),
                "header": int(bool(w) and w <= hdr),
                "header_content": len(info & hdr) / max(1, len(info)),
                "pk": int(n in pk),
                "fk": int(n in fk),
                "idword": int("id" in w),
                "owners": log1p(len(owners.get(n, ()))),
                "short": int(len(w) == 1),
            }
            if role != "table":
                val["table_named"] = int(
                    any(
                        bool(stem(t) - IGNORE) and (stem(t) - IGNORE) <= qw
                        for t in owners[n]
                    )
                )
                val["table_hinted"] = sum(
                    bool(owners[n] & z["table"]) for z in src[1:]
                ) / len(hints)
            else:
                val["owns_header"] = int(
                    any(n in ts and stem(c) <= hdr for c, ts in owners.items())
                )
            for k in ("coverage", "header", "pk", "focus", "all_words", "table_named"):
                if val.get(k):
                    val["hint/" + k] = hs * val[k]
                    val["draft/" + k] = draft * val[k]
            out[role][n] = {k: v for k, v in val.items() if v}
    return out


def predict(slots, model):
    return {
        r: {
            n: sum(model[r].get(k, 0) * v for k, v in x.items())
            for n, x in domain.items()
        }
        for r, domain in slots.items()
    }


def evidence(roles, prob):
    out = {}
    for r, chosen in roles.items():
        p = prob[r]
        inside = [max(-4, min(4, p.get(n, -5))) for n in chosen]
        out["link/" + r + "/sum"] = max(-7, min(7, sum(inside)))
        out["link/" + r + "/weakest"] = min(inside, default=0)
        out["link/" + r + "/missing"] = max(
            (min(4, v) for n, v in p.items() if n not in chosen and v > 0), default=0.0
        )
    return out
