"""One public action at a time. The search policy and final decision are independently switchable."""

from pathlib import Path
from priority import Agent as Priority
from repair.controller import load_config
from repair.search import canonical
from repair.preferences import feedback
from evidence import (
    expand_features,
    forest_score,
    load_policy,
    ref_library,
    reference_context,
)
from feedback_search import implication


class Agent(Priority):
    def build(self, obs):
        pool = super().build(obs)
        if (
            self.cfg.get("tree_search")
            or self.cfg.get("tree_final")
            or self.cfg.get("belief")
        ):
            g = reference_context(obs, ref_library())
            self.fx = {}
            self.pr = {}
            for _, sql in pool.values():
                k = canonical(sql)
                self.fx[k] = expand_features(self.raw_features[k], sql, obs, g)
                self.pr[k] = forest_score(self.fx[k], load_policy()["search"])
        if self.cfg.get("tree_search"):
            order_old = sorted(
                pool, key=lambda k: (pool[k][0], pool[k][1]), reverse=True
            )
            order_new = sorted(
                order_old, key=lambda k: self.pr[canonical(pool[k][1])], reverse=True
            )
            ranks = {k: i for i, k in enumerate(order_old)}
            pool = {
                k: (1 / (2 + ranks[k]) + 1.6 / (2 + i), pool[k][1])
                for i, k in enumerate(order_new)
            }
        return pool

    def act(self, obs):
        if self.cfg.get("staged"):
            from staged import act

            return act(self, obs)
        action = super().act(obs)
        if len(obs["history"]) < 3:
            return action
        checks = [e for e in obs["history"] if e["tool"] == "check"]
        used = {canonical(e["sql"]) for e in checks}
        if (
            self.cfg.get("diagnose")
            and obs["remaining"]
            and self.ground_query is None
            and action["tool"] != "execute"
        ):
            from live_repair import action as repair_action

            diagnostic = repair_action(self, obs, checks)
            if diagnostic:
                return diagnostic
        if self.cfg.get("belief"):
            from belief import state, next_check, choose

            post = state(self, obs, checks)
            can_schedule = (
                action["tool"] != "execute" and obs["history"][-1]["tool"] != "execute"
            )
            if self.cfg.get("late"):
                # Spend the first two post-schema checks on the inherited order.
                can_schedule = can_schedule and len(checks) >= 3
            if (
                self.cfg.get("reserve")
                and len(checks) < 3
                and action["tool"] != "submit"
            ):
                can_schedule = False
            if obs["remaining"] and can_schedule:
                proposal = next_check(self, obs, checks, post)
                if proposal:
                    return dict(tool="check", sql=proposal)
            if action["tool"] == "submit":
                selected = choose(self, obs, checks, action, post)
                if self.cfg.get("role_final"):
                    passing = [
                        e["sql"]
                        for e in checks
                        if e["observation"]["matches_example"]
                        and canonical(e["sql"]) in self.roles
                    ]
                    incumbent = selected["sql"]
                    if incumbent in passing:
                        support = {s: post.get(s, 0.0) for s in passing}
                        from math import log

                        def utility(s):
                            k = canonical(s)
                            return (
                                log(support[s] + 0.002)
                                + 0.35 * self.roles[k]
                                + 0.16 * self.pr[k]
                            )

                        best = max(
                            passing, key=lambda s: (utility(s), s == incumbent, s)
                        )
                        if utility(best) > utility(incumbent) + 0.55:
                            selected = dict(tool="submit", sql=best)
                return selected
        if (
            self.cfg.get("spend")
            and obs["remaining"]
            and action["tool"] != "execute"
            and obs["history"][-1]["tool"] != "execute"
        ):
            todo = []
            for s, sql in self.pool.values():
                if canonical(sql) in used:
                    continue
                effect = implication(sql, obs, checks) if self.cfg.get("infer") else 0
                if effect >= 0:
                    todo.append((effect, s, sql))
            if todo:
                return dict(tool="check", sql=max(todo)[2])
        if action["tool"] != "submit" or not self.cfg.get("tree_final"):
            return action
        incumbent = action["sql"]
        passing = [e for e in checks if e["observation"]["matches_example"]]
        k = canonical(incumbent)
        if len(passing) < 2 or k not in self.fx:
            return action
        scores = {}
        for e in passing:
            sql = e["sql"]
            key = canonical(sql)
            if key in self.fx:
                x = self.fx[key] | feedback(obs, e["observation"])
                scores[sql] = forest_score(x, load_policy()["final"])
        if incumbent not in scores:
            return action
        best = max(scores, key=lambda s: (scores[s], s == incumbent, s))
        if scores[best] - scores[incumbent] > self.cfg.get("switch", 0.25):
            return dict(tool="submit", sql=best)
        return action


def create_agent(checkpoint=None):
    return Agent(load_config(checkpoint or Path(__file__).with_name("config.json")))
