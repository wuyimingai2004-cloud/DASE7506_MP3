"""Estimate semantic roles of qualified schema columns using train-reference supervision."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json
import math
import re
from sqlglot import exp
from repair.edits import parse, tables
from repair.preferences import stem
from transforms import sources

ROLES = ("table", "expressions", "where", "group", "order", "having", "join")
MISSING = ("", "")


def pieces(q):
    # Separate requested attributes from the subject, constraint and sorting phrases.
    chunks = re.split(
        r"\b(?:who(?:se)?|where|that|which have|with|such that|having|ordered by|sorted by|order by|sort by)\b",
        q,
        flags=re.I,
    )
    return stem(chunks[0]), stem(" ".join(chunks[1:]))


def catalog(o):
    return {r["name"].lower(): r for r in o["history"][2]["observation"]}


def positions(tree, cat):
    facts = {r: set() for r in ROLES}
    if tree is None:
        return facts
    for s in tree.find_all(exp.Select):
        env = tables(s)
        facts["table"].update((t, "") for t in env.values())
        for key in ROLES[1:]:
            v = s.args.get("joins" if key == "join" else key) or []
            for part in v if isinstance(v, list) else [v]:
                for c in part.find_all(exp.Column):
                    if c.find_ancestor(exp.Select) is not s:
                        continue
                    name = c.name.lower()
                    own = env.get(c.table.lower())
                    if not own:
                        cand = [
                            t
                            for t in env.values()
                            if any(
                                m["name"].lower() == name
                                for m in cat.get(t, {}).get("columns", ())
                            )
                        ]
                        if len(cand) != 1:
                            continue
                        own = cand[0]
                    facts[key].add((own, name))
    return facts


def overlap(ws, qs):
    return len(ws & qs) / max(1, len(ws))


def features(o, cat):
    q = o["task"]["question"].lower()
    qw = stem(q)
    head, tail = pieces(q)
    headers = stem(" ".join(o["example"]["columns"]))
    hint_sql = o["history"][1]["observation"]
    src = [positions(parse(s), cat) for s in [o["issue_sql"], *hint_sql]]
    near_literals = set()
    for s in sources(o):
        t = parse(s)
        if t is None:
            continue
        for lit in t.find_all(exp.Literal):
            if not lit.this:
                continue
            pos = q.find(lit.this.lower())
            if pos >= 0:
                near_literals.update(stem(q[max(0, pos - 22) : pos]))
    ow = Counter(m["name"].lower() for r in cat.values() for m in r["columns"])
    fs = defaultdict(dict)
    for t, r in cat.items():
        tw = stem(t)
        fk = {v["from"].lower() for v in r["foreign_keys"] if v["from"]}
        for info in [None, *r["columns"]]:
            n = info["name"].lower() if info else ""
            nw = stem(n)
            fact = (t, n)
            for role in ROLES[1:] if info else ("table",):
                hint = sum(fact in s[role] for s in src[1:]) / len(hint_sql)
                hinted_table = sum((t, "") in s["table"] for s in src[1:]) / len(
                    hint_sql
                )
                other_support = sum(
                    any(fact in slots for slots in s.values()) for s in src[1:]
                ) / len(hint_sql)
                x = dict(
                    bias=1.0,
                    tq=overlap(tw, qw),
                    nq=overlap(nw, qw),
                    t_head=overlap(tw, head),
                    n_head=overlap(nw, head),
                    t_tail=overlap(tw, tail),
                    n_tail=overlap(nw, tail),
                    header=overlap(nw, headers),
                    literal=overlap(nw, near_literals),
                    hint=hint,
                    unanimous=int(hint == 1),
                    mixed=int(0 < hint < 1),
                    draft=int(fact in src[0][role]),
                    table_hint=hinted_table,
                    any_hint=other_support,
                    n_unique=int(ow[n] == 1),
                    t_full=int(tw <= qw),
                    n_full=int(bool(nw) and nw <= qw),
                )
                if n:
                    x.update(
                        pk=int(bool(info["pk"])),
                        fk=int(n in fk),
                        id=int("id" in nw),
                        numeric=int(
                            bool(re.search("int|num|real|float", info["type"], re.I))
                        ),
                    )
                    x["owner_and_col"] = x["tq"] * x["nq"]
                    x["owner_and_header"] = x["tq"] * x["header"]
                    x["absent_owner"] = int(
                        n in ow and not hinted_table and not (tw & qw)
                    )
                else:
                    x["owns"] = sum(
                        c["name"].lower()
                        in [h.lower() for h in o["example"]["columns"]]
                        for c in r["columns"]
                    )
                    x["fk_count"] = min(4, len(fk))
                for k in (
                    "tq",
                    "nq",
                    "header",
                    "pk",
                    "fk",
                    "id",
                    "table_hint",
                    "draft",
                ):
                    if x.get(k):
                        x["supported/" + k] = x[k] * hint
                fs[role][fact] = {k: min(4, v) for k, v in x.items() if v}
    return fs


@lru_cache(maxsize=1)
def model():
    return json.loads(Path(__file__).with_name("roles.json").read_text())


def estimate(o):
    cat = catalog(o)
    xs = features(o, cat)
    w = model()
    return cat, {
        r: {k: sum(w[r].get(j, 0) * v for j, v in x.items()) for k, x in group.items()}
        for r, group in xs.items()
    }


def score(sql, ctx):
    cat, probs = ctx
    chosen = positions(parse(sql), cat)
    factors = dict(
        table=1.2,
        expressions=1.2,
        where=1.0,
        group=0.45,
        order=0.7,
        having=0.3,
        join=0.36,
    )
    return sum(
        factors[r] * sum(max(-4.0, min(4.0, probs[r].get(k, -6))) for k in vals)
        for r, vals in chosen.items()
    )
