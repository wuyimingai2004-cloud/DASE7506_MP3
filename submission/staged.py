"""Spend extra checks only after preserving a decision reached by the inherited policy."""

from collections import Counter
from priority import Agent as Priority
from repair.search import canonical
from repair.edits import parse
from repair.preferences import feedback
from evidence import forest_score, load_policy
from belief import state, next_check, condition
from arbitrate import evidence, preference


def baseline(agent):
    if not hasattr(agent, "first"):
        cfg = agent.cfg.copy()
        cfg.update(augment=False, compiler=False, tree_final=False, tree_search=False)
        if cfg.get("committee_first"):
            cfg.update(priority=False, explore=True, duel=True, vote=True)
        agent.first = Priority(cfg)
    return agent.first


def resolve(agent, o, checks, post):
    passing = [e for e in checks if e["observation"]["matches_example"]]
    if not passing:
        return dict(tool="submit", sql=agent.inc)
    incumbent = next((e for e in passing if e["sql"] == agent.inc), None)
    available = [e for e in passing if canonical(e["sql"]) in agent.fx]
    if not available or (incumbent and incumbent not in available):
        return dict(
            tool="submit", sql=incumbent["sql"] if incumbent else passing[0]["sql"]
        )

    def facts(e):
        s = e["sql"]
        k = canonical(s)
        header = feedback(o, e["observation"])
        return (
            forest_score(agent.fx[k] | header, load_policy()["final"]),
            post.get(s, 0),
            agent.verified[k],
            agent.local[k][1],
            evidence(s, o, e["observation"], agent.sources),
        )

    scored = [(e["sql"], facts(e)) for e in available]
    if incumbent:
        inc = incumbent["sql"]
    else:
        inc = max(scored, key=lambda z: (z[1][2], z[0]))[0]
    if agent.cfg.get("keep") and incumbent:
        return dict(tool="submit", sql=inc)
    base = dict(scored)[inc]
    choices = []
    for s, x in scored:
        if s == inc:
            continue
        pair = preference(s, inc, x[-1], base[-1], o)
        if agent.cfg.get("arbiter") == "pair":
            if pair > 0.15:
                choices.append((pair, s))
        else:
            votes = sum(
                (
                    x[0] > base[0] + 0.1,
                    x[1] > base[1] * 1.3,
                    x[2] > base[2] + 0.4,
                    x[3] > base[3] + 0.4,
                )
            )
            if (votes >= 3 and pair >= -0.04) or (votes >= 2 and pair > 0.18):
                choices.append((votes + 2 * pair + 0.1 * (x[0] - base[0]), s))
    return dict(tool="submit", sql=max(choices)[1] if choices else inc)


def act(agent, o):
    first = baseline(agent)
    h = o["history"]
    if len(h) < 3:
        return first.act(o)
    if len(h) == 3:
        agent.initial_observation = o
    checks = [e for e in h if e["tool"] == "check"]
    passing = [e for e in checks if e["observation"]["matches_example"]]
    if not hasattr(agent, "inc"):
        action = first.act(o)
        weak = condition(o) in ("empty", "nullzero")
        if action["tool"] == "submit":
            if (
                action["sql"] in {e["sql"] for e in passing}
                and not weak
                and not agent.cfg.get("audit_strong")
            ):
                return action
            if not o["remaining"] and not agent.cfg.get("arbiter"):
                return action
            agent.inc = action["sql"]
        elif (
            o["remaining"] <= agent.cfg.get("reserve_slots", 3)
            and not passing
            and action["tool"] != "execute"
            and h[-1]["tool"] != "execute"
        ):
            agent.inc = max(
                checks,
                key=lambda e: (
                    e["observation"]["ok"],
                    first.details.get(canonical(e["sql"]), -30),
                ),
            )["sql"]
        else:
            return action
    if agent.pool is None:
        agent.pool = agent.build(agent.initial_observation)
    # Initial model construction uses only the real, shared first three observations.
    post = state(agent, o, checks)
    if o["remaining"]:
        s = next_check(agent, o, checks, post)
        if s:
            return dict(tool="check", sql=s)
    return resolve(agent, o, checks, post)
