"""Interleave independent candidate orders and use failures to avoid redundant checks."""

from functools import lru_cache
from sqlglot import exp
from repair.edits import parse
from repair.preferences import family
from repair.search import canonical


@lru_cache(maxsize=2048)
def row_source(sql):
    t = parse(canonical(sql))
    if not isinstance(t, exp.Select) or any(
        t.args.get(k)
        for k in ("group", "distinct", "having", "limit", "offset", "with")
    ):
        return None
    if any(p.find(exp.AggFunc) or p.find(exp.Subquery) for p in t.expressions):
        return None
    t = t.copy()
    t.set("expressions", [exp.Literal.number(1)])
    t.set("order", None)
    return t.sql(dialect="sqlite")


def contradicted(sql, obs, failed):
    n = obs["example"]["row_count"]
    for e in failed:
        fb = e["observation"]
        if not fb["ok"]:
            continue
        if fb["row_count"] != n:
            key = row_source(sql)
            if key and key == row_source(e["sql"]):
                return True
        if not obs["ordered"] and (fb["row_count"] == 0 or n == 0):
            key = family(sql)
            if key and key == family(e["sql"]):
                return True
    return False


def next_candidate(agent, obs):
    checks = [e for e in obs["history"] if e["tool"] == "check"]
    seen = {canonical(e["sql"]) for e in checks}
    failed = [e for e in checks if not e["observation"]["matches_example"]]
    pool = [
        sql for _, sql in agent.original_pool.values() if canonical(sql) not in seen
    ]
    if not pool:
        return
    choices = [sql for sql in pool if not contradicted(sql, obs, failed)]
    if not choices:
        return
    # Fixed slot schedule. Preserve the independent hint as a competitor to generated edits.
    used = len(checks) - 1

    def key(sql):
        k = canonical(sql)
        if used == 0:
            return agent.sources.get(k, 0), agent.details[k], agent.local[k][0], sql
        if used % 2:
            return agent.local[k][0], agent.details[k], sql
        return agent.details[k], agent.local[k][0], sql

    if (
        agent.cfg.get("conditional")
        and used
        and (agent.cfg["conditional"] == "all" or used % 2)
    ):
        from repair.conditional import estimate

        # Search over the union of shortlists. A query can enter from language, the baseline,
        # or a supported source. The latest public discrepancy chooses within the union.
        ranks = [
            lambda s: agent.details[canonical(s)],
            lambda s: agent.local[canonical(s)][0],
        ]
        shortlist = set()
        for metric in ranks:
            shortlist.update(
                sorted(choices, key=lambda s: (metric(s), s), reverse=True)[:4]
            )
        shortlist.update(s for s in choices if agent.sources.get(canonical(s)))
        return max(
            shortlist,
            key=lambda s: (
                estimate(s, agent.raw_features[canonical(s)], obs, checks),
                key(s),
            ),
        )
    return max(choices, key=key)
