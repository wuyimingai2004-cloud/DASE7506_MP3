from pathlib import Path
from functools import lru_cache
from sqlglot import exp
from semantic_priority import Agent as Priority
from repair.controller import load_config
from repair.edits import parse
from repair.search import canonical
from semplan import plans
from predicate_scope import comparison


@lru_cache(maxsize=1024)
def structure(sql):
    t = parse(sql)
    if t is None:
        return set(), set(), set(), set()
    tables = {x.name.lower() for x in t.find_all(exp.Table)}
    columns = {x.name.lower() for x in t.find_all(exp.Column)}
    flags = set()
    if t.find(exp.Having) or (t.find(exp.Count) and t.find(exp.Subquery)):
        flags.add("count_group")
    if (
        t.find(exp.Except)
        or (t.find(exp.Not) and (t.find(exp.Subquery) or t.find(exp.Exists)))
        or (any(j.side == "LEFT" for j in t.find_all(exp.Join)) and t.find(exp.Is))
    ):
        flags.add("anti")
    if t.find(exp.Intersect):
        flags.add("intersection")
    if t.find(exp.Union) or t.find(exp.Or):
        flags.add("disjunction")
    if t.find(exp.Except) or (
        t.find(exp.Not) and (t.find(exp.Subquery) or t.find(exp.Exists))
    ):
        flags.add("group_exclusion")
    if t.find(exp.Join):
        flags.add("role_join")
    literals = {str(x.this).casefold() for x in t.find_all(exp.Literal)}
    return tables, columns, flags, literals


def agreement(plan, sql):
    tables, columns, flags, literals = structure(sql)
    required = {t.lower() for t in plan.tables}
    keys = {k.lower() for k in plan.keys}
    values = structure(plan.sql)[3]
    return (
        int(plan.family in flags),
        len(tables & required) / len(required),
        len(columns & keys) / len(keys) if keys else 1.0,
        len(values & literals) / len(values) if values else 1.0,
    )


def compatible(plan, sql):
    exact = comparison(plan, sql)
    return exact if exact is not None else all(x == 1 for x in agreement(plan, sql))


def contradicts(plan, sql):
    exact = comparison(plan, sql)
    if exact is not None:
        return not exact
    flags = structure(sql)[2]
    # A correlated subquery, grouping, or HAVING can express the same entity set.
    # Preserve these alternative formulations when their binding cannot be proven.
    tree = parse(sql)
    if (
        not tree
        or any(tree.find(op) for op in (exp.Having, exp.Subquery, exp.Exists))
        or plan.family in flags
    ):
        return False
    return plan.family in (
        "intersection",
        "disjunction",
        "group_exclusion",
        "anti",
        "count_group",
    )


class Agent(Priority):
    def semantic_plans(self, obs, schema):
        self.plans = plans(obs["task"]["question"], obs["example"]["columns"], schema)
        return self.plans

    def act(self, obs):
        action = super().act(obs)
        if self.cfg["variant"] == "generate" or not getattr(self, "plans", None):
            return action
        history = obs["history"]
        checked = {canonical(e["sql"]): e for e in history if e["tool"] == "check"}
        passing = [
            e["sql"] for e in checked.values() if e["observation"]["matches_example"]
        ]
        if getattr(self, "reserved", None):
            # A failed reserved query cannot displace an incumbent. For two public
            # matches the grounded chooser keeps its decision unless it contradicts
            # a bound role and the challenger fits that role.
            if action["tool"] == "submit":
                answer = self.reserved
                incumbents = [s for s in passing if s != answer]
                if answer in passing:
                    if not incumbents:
                        return {"tool": "submit", "sql": answer}
                    # An operator-level contradiction is necessary to replace an
                    # already passing query. Other binding/grain choices are ambiguous.
                    conflicts = all(contradicts(self.target, s) for s in incumbents)
                    if compatible(self.target, answer) and conflicts:
                        return {"tool": "submit", "sql": answer}
                if action["sql"] == answer:
                    if incumbents:
                        original = self.displaced
                        if (
                            original["tool"] == "submit"
                            and original["sql"] in incumbents
                        ):
                            return original
                        return {
                            "tool": "submit",
                            "sql": max(
                                incumbents,
                                key=lambda s: self.verified.get(canonical(s), -30),
                            ),
                        }
                    if answer not in passing:
                        old = [e for e in checked.values() if e["sql"] != answer]
                        chosen = max(
                            old,
                            key=lambda e: (
                                e["observation"]["ok"],
                                self.details.get(canonical(e["sql"]), -30),
                                e["sql"],
                            ),
                        )
                        return {"tool": "submit", "sql": chosen["sql"]}
            return action
        if (
            not obs["remaining"]
            or action["tool"] == "execute"
            or (history[-1]["tool"] == "execute" and action["tool"] == "check")
        ):
            return action
        if action["tool"] == "check" and (len(checked) < 2 or obs["remaining"] > 2):
            return action
        options = [p for p in self.plans if canonical(p.sql) not in checked]
        if not options:
            return action
        plan = options[0]
        if passing and not all(contradicts(plan, s) for s in passing):
            return action
        # If the next ordinary check already represents the plan, give it the slot.
        if action["tool"] == "check":
            if plan.family in ("intersection", "disjunction", "group_exclusion"):
                represents_plan = comparison(plan, action["sql"]) is True
            else:
                represents_plan = compatible(plan, action["sql"])
            if represents_plan:
                return action
        self.gate = True
        if self.cfg["variant"] == "oldslot":
            options = [
                (compatible(plan, sql), agreement(plan, sql), score, sql)
                for score, sql in self.pool.values()
                if canonical(sql) not in checked
            ]
            if not options:
                return action
            answer = max(options)[-1]
        else:
            answer = plan.sql
        self.reserved = answer
        self.target = plan
        self.displaced = action
        return {"tool": "check", "sql": answer}


def create_agent(checkpoint=None):
    return Agent(load_config(checkpoint or Path(__file__).with_name("config.json")))
