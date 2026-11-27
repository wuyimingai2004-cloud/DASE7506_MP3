from collections import Counter
from pathlib import Path
from semantic_grounded import Agent as Grounded
from repair.controller import load_config
from repair.search import canonical
from repair.preferences import context, features, feedback, score, weights


class Agent(Grounded):
    def entries(self, obs, neighbors=None):
        records = super().entries(obs, neighbors)
        self.records = records
        return records

    def build(self, obs):
        pool = super().build(obs)
        self.original_pool = pool
        self.local = {}
        if not any(self.cfg.get(k) for k in ("priority", "choice", "explore", "duel")):
            return pool
        ctx = context(obs)
        model = weights()
        self.sources = Counter(
            canonical(sql)
            for sql in next(
                e["observation"] for e in obs["history"] if e["tool"] == "repair_hint"
            )
        )
        self.raw_features = {}
        for sql, _, raw in self.records:
            x = features(sql, raw, obs, ctx)
            self.raw_features[canonical(sql)] = x
            self.local[canonical(sql)] = (
                score(x, model["search"]),
                score(x, model["final"]),
            )
        if not self.cfg.get("priority"):
            return pool
        return {k: (self.local[canonical(sql)][0], sql) for k, (_, sql) in pool.items()}

    def act(self, obs):
        action = super().act(obs)
        if len(obs["history"]) >= 3 and self.cfg.get("explore") and obs["remaining"]:
            static_keys = self.local
            if (
                action["tool"] == "submit"
                and self.cfg.get("rescue")
                and not hasattr(self, "incumbent")
            ):
                self.incumbent = action["sql"]
            if action["tool"] == "submit" or (
                (not self.cfg.get("rescue") or hasattr(self, "incumbent"))
                and action["tool"] == "check"
                and canonical(action["sql"]) in static_keys
                and obs["history"][-1]["tool"] != "execute"
            ):
                from repair.exploration import next_candidate

                proposal = next_candidate(self, obs)
                if proposal:
                    return {"tool": "check", "sql": proposal}
        mode = self.cfg.get("choice") or self.cfg.get("duel")
        if action["tool"] != "submit" or not mode or not getattr(self, "local", None):
            return action
        passing = [
            e
            for e in obs["history"]
            if e["tool"] == "check" and e["observation"]["matches_example"]
        ]
        if len(passing) < 2:
            return action
        if self.cfg.get("duel"):
            return self.tournament(obs, action, passing)
        options = {}
        for e in passing:
            k = canonical(e["sql"])
            if k not in self.local:
                # Keep a live value-probe winner when its parent has no static representation.
                if e["sql"] == action["sql"]:
                    return action
                continue
            options[e["sql"]] = self.local[k][1] + score(
                feedback(obs, e["observation"]), weights()["final"]
            )
        if action["sql"] in options:
            best = max(
                options, key=lambda sql: (options[sql], sql == action["sql"], sql)
            )
            if options[best] - options[action["sql"]] > self.cfg.get(
                "switch_margin", 0
            ):
                return {"tool": "submit", "sql": best}
        return action

    def tournament(self, obs, action, passing):
        from repair.duel import candidate, difference, parameters
        from repair.linker import predict, slot_inputs

        par = parameters()
        if canonical(action["sql"]) not in self.raw_features:
            return action
        prob = predict(slot_inputs(obs), par["linker"])
        represented = {}
        for e in passing:
            k = canonical(e["sql"])
            if k in self.raw_features:
                x = self.raw_features[k] | feedback(obs, e["observation"])
                represented[e["sql"]] = candidate(e["sql"], x, prob)
        incumbent = action["sql"]
        if len(represented) < 2:
            return action
        # A switch must defeat every checked passing alternative in a direct comparison.
        # Ties, cycles and inconclusive evidence preserve the original grounded decision.
        margin = self.cfg.get("duel_margin", 1.0)
        if self.cfg.get("vote"):
            from math import tanh

            # Repeated close substitutes can veto one another in a unanimity tournament.
            # Compare unordered pairs once, and require a direct win against the incumbent.
            ballots = dict.fromkeys(represented, 0.0)
            direct = {}
            for i, (sa, a) in enumerate(represented.items()):
                for sb in list(represented)[i + 1 :]:
                    v = score(difference(a, represented[sb]), par["pair"])
                    ballots[sa] += tanh(v / 2)
                    ballots[sb] -= tanh(v / 2)
                    direct[sa, sb] = v
                    direct[sb, sa] = -v
            best = max(ballots, key=lambda s: (ballots[s], s == incumbent, s))
            if best != incumbent and direct[best, incumbent] > margin:
                return {"tool": "submit", "sql": best}
            return action
        for challenger, a in represented.items():
            if challenger != incumbent and all(
                score(difference(a, b), par["pair"]) > margin
                for sql, b in represented.items()
                if sql != challenger
            ):
                return {"tool": "submit", "sql": challenger}
        return action


def create_agent(checkpoint=None):
    return Agent(load_config(checkpoint or Path(__file__).with_name("config.json")))
