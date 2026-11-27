"""Learn when two candidate programs collide on a public result. Update hypotheses after real checks."""

import json
import math
from functools import lru_cache
from pathlib import Path
from sqlglot import exp
from repair.edits import core, emit, parse, main_select
from repair.search import canonical
from repair.preferences import properties
from feedback_search import implication


@lru_cache(maxsize=8192)
def descriptor(sql):
    key = canonical(sql)
    t = parse(key)
    if t is None:
        return ("invalid",) * 12
    s = main_select(t)

    def val(k):
        n = s.args.get(k)
        return emit(n) if n is not None else ""

    return (
        t.key,
        tuple(core(n).sql() for n in s.expressions),
        val("distinct"),
        tuple(sorted(n.name.lower() for n in t.find_all(exp.Table))),
        tuple(sorted(emit(j) for j in t.find_all(exp.Join))),
        val("where"),
        val("group"),
        val("having"),
        val("order"),
        val("limit"),
        frozenset(n.this for n in t.find_all(exp.Literal)),
        tuple(n.key for n in t.find_all(exp.AggFunc)),
    )


@lru_cache(maxsize=32768)
def difference(a, b):
    x = descriptor(a)
    y = descriptor(b)
    return sum(1 << i for i, (u, v) in enumerate(zip(x, y)) if u != v)


def condition(o):
    rows = o["example"]["rows"]
    if rows == []:
        return "empty"
    if rows == [[0]] or rows == [[None]]:
        return "nullzero"
    if o["example"]["row_count"] == 1:
        return "single"
    return "many"


def group_keys(mask, kind):
    # Hierarchical shrinkage, first by result size, then broad edit, then exact edit.
    return [
        kind,
        kind + "/" + str(mask.bit_count()),
        kind + "/m" + str(mask & 0x3EF),
        str(mask),
        kind + "/" + str(mask) + "/exact",
    ]


@lru_cache(maxsize=1)
def parameters():
    return json.loads(Path(__file__).with_name("collisions.json").read_text())


@lru_cache(maxsize=8192)
def predict_pair(mask, kind):
    p = 0.40
    lookup = parameters()
    for key in group_keys(mask, kind):
        hit, n = lookup.get(key, (0, 0))
        p = (hit + 8 * p) / (n + 8)
    return max(0.035, min(0.97, p))


def initial(agent, o):
    sqls = list(dict.fromkeys(s for _, s in agent.pool.values()))
    orders = [
        lambda s: agent.local[canonical(s)][0],
        lambda s: agent.verified[canonical(s)],
        lambda s: agent.pr[canonical(s)],
    ]
    weights = [0.36, 0.36, 0.28]
    if agent.cfg.get("role_search") or agent.cfg.get("role_final"):
        from roles import estimate, score

        ctx = estimate(o)
        agent.roles = {canonical(s): score(s, ctx) for s in sqls}
        if agent.cfg.get("role_search"):
            orders.append(lambda s: agent.roles[canonical(s)])
            weights = [0.27, 0.27, 0.22, 0.24]
    priors = dict.fromkeys(sqls, 0.0)
    for metric, weight in zip(orders, weights):
        scores = sorted(sqls, key=lambda s: (-metric(s), s))
        for i, s in enumerate(scores):
            priors[s] += weight / (1 + i) ** 1.6
    return priors


def state(agent, o, checks):
    if not hasattr(agent, "hypotheses"):
        agent.hypotheses = initial(agent, o)
    kind = condition(o)
    post = dict(agent.hypotheses)
    for e in checks:
        s = e["sql"]
        passed = e["observation"]["matches_example"]
        canon = canonical(s)
        for h in post:
            if canonical(h) == canon:
                post[h] *= 1.0 if passed else 0.0
            else:
                p = predict_pair(difference(h, s), kind)
                # Related observations are not independent; temper repeated evidence.
                post[h] *= (p if passed else 1 - p) ** 0.65
    mass = sum(post.values()) or 1.0
    return {s: p / mass for s, p in post.items()}


def next_check(agent, o, checks, post):
    kind = condition(o)
    tried = {canonical(e["sql"]) for e in checks}
    passing = any(e["observation"]["matches_example"] for e in checks)
    # Bound the likelihood calculation, retaining the bulk of posterior mass.
    ranked = sorted(post, key=lambda s: -post[s])
    top = ranked[:12]
    choices = [
        s
        for s in ranked
        if canonical(s) not in tried and implication(s, o, checks) >= 0
    ][:12]
    if not choices:
        return

    def score(s):
        probs = [1.0 if s == h else predict_pair(difference(h, s), kind) for h in top]
        mass = sum(post[h] for h in top) or 1.0
        pred = sum(p * post[h] for p, h in zip(probs, top)) / mass
        # Expected variance of the binary check likelihood over the surviving programs.
        separation = sum(post[h] * (p - pred) ** 2 for p, h in zip(probs, top)) / mass
        return post[s] + (0.65 if passing else 0.25) * separation + 0.03 * pred, s

    return max(choices, key=score)


def choose(agent, o, checks, action, post):
    if not agent.cfg.get("belief_final"):
        return action
    pool = {canonical(s): s for s in post}
    passing = [e for e in checks if e["observation"]["matches_example"]]
    opts = {
        e["sql"]: post[pool[canonical(e["sql"])]]
        for e in passing
        if canonical(e["sql"]) in pool
    }
    inc = action["sql"]
    if inc not in opts or len(opts) < 2:
        return action
    best = max(opts, key=lambda s: (opts[s], s == inc, s))
    if (
        opts[best] > 2.0 * opts[inc]
        and agent.pr[canonical(best)] > agent.pr[canonical(inc)]
    ):
        return dict(tool="submit", sql=best)
    return action
