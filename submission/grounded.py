"""Offline feedback controller and TRAIN-fitted candidate selector."""

from pathlib import Path
from collections import Counter
from repair.controller import Agent as Legacy, load_config
from repair.edits import emit, parse, main_select, projection_distance
from repair.relational import Catalog
from repair.sketch import retrieve, sketch_tokens, templates
from repair.search import canonical, dot, edits, model, values


class Agent(Legacy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.details = {}
        self.verified = {}
        self.ground_query = None
        self.legacy = None
        self.switched = False

    def entries(self, obs, neighbors=None):
        base = super().build(obs)
        self.legacy = base
        q = obs["task"]["question"].lower()
        example = obs["example"]
        schema = next(e["observation"] for e in obs["history"] if e["tool"] == "schema")
        cat = Catalog(schema)
        sources = [obs["issue_sql"]] + list(
            next(e["observation"] for e in obs["history"] if e["tool"] == "repair_hint")
        )
        votes = Counter(canonical(sql) for sql in sources[1:])
        draft_key = canonical(sources[0])
        if neighbors is None:
            neighbors = retrieve(sketch_tokens(q, example["columns"], cat))
        rows = {}

        def add(sql, prior, kind, support=0):
            if len(sql.encode()) > 16000:
                return
            key = canonical(sql)
            if key not in rows:
                rows[key] = [sql, prior, {}, support]
            r = rows[key]
            if prior > r[1]:
                r[:2] = [sql, prior]
            r[2][kind] = 1
            r[3] = max(support, r[3])

        for score, sql in base.values():
            add(sql, score, "legacy")
        if self.cfg.get("expand", True):
            seen = set()
            for sql in sources:
                key = canonical(sql)
                if key in seen:
                    continue
                seen.add(key)
                t = parse(sql)
                if t is None:
                    continue
                score = base.get(emit(t), (0, None))[0]
                d = min(4, projection_distance(t, example["columns"]))
                for variant, tag in edits(t, q, example, cat):
                    prior = (
                        score
                        + 2
                        * (d - min(4, projection_distance(variant, example["columns"])))
                        - 1.4
                    )
                    add(emit(variant), prior, tag)
                if self.cfg.get("templates", True):
                    for variant, support in templates(t, q, example, cat, neighbors):
                        add(
                            emit(variant),
                            score + 2 * d - 1.8 + support,
                            "template",
                            support,
                        )
            if self.cfg.get("synthesis"):
                from repair.synthesis import generate

                trees = list(
                    {
                        canonical(sql): parse(sql)
                        for sql in sources
                        if parse(sql) is not None
                    }.values()
                )
                reference = max((score for score, _ in base.values()), default=0)
                for variant, label, confidence in generate(trees, obs, cat, schema):
                    if variant is not None:
                        add(emit(variant), reference - 2.5 + confidence, label)
        if self.cfg.get("augment"):
            from transforms import edits as transferred, tags
            from graph_edits import variants
            context = tags(obs, schema)
            reference = max((r[1] for r in rows.values()), default=0)
            for source in dict.fromkeys(sources):
                tree = parse(source)
                if tree is None:
                    continue
                proposals = list(transferred(tree, context)) + list(variants(tree, obs, cat))
                for candidate, kind, support in proposals:
                    sql = emit(candidate)
                    if canonical(sql) not in rows:
                        add(sql, reference - 2.4 + support, kind)
        if self.cfg.get("compiler"):
            from compiler import generate as instantiate
            reference = max((r[1] for r in rows.values()), default=0)
            for candidate, kind, support in instantiate(obs):
                sql = emit(candidate)
                if canonical(sql) not in rows:
                    add(sql, reference - 2.4 + support, kind)
        highest = max((r[1] for r in rows.values()), default=0)
        records = []
        for key, (sql, prior, tags, support) in rows.items():
            tree = parse(sql)
            if tree is None:
                continue
            features = values(tree, q, example, cat, neighbors)
            features.update(
                {
                    "prior": max(-12, prior - highest),
                    "votes": votes[key] / 4,
                    "draft": int(key == draft_key),
                    "template_support": support,
                    **{"gen_" + kind: v for kind, v in tags.items()},
                }
            )
            records.append((sql, prior, features))
        return records

    def build(self, obs):
        if not self.cfg.get("policy"):
            return super().build(obs)
        pool = {}
        intent = None
        if self.cfg.get("intent"):
            from repair.intent import context, logits, parameters, fidelity

            schema = next(
                e["observation"] for e in obs["history"] if e["tool"] == "schema"
            )
            intent = logits(context(obs, Catalog(schema)), parameters())
        entries = self.entries(obs)
        legacy_order = {}
        if self.cfg.get("protect"):
            for _, sql in sorted(self.legacy.values(), key=lambda item: -item[0]):
                legacy_order.setdefault(canonical(sql), len(legacy_order))
        for sql, prior, x in entries:
            score = prior
            if self.cfg.get("rerank", True):
                score = self.cfg.get("prior_mix", 0.4) * x["prior"] + dot(x, model())
            if self.cfg.get("judge"):
                from repair.judge import input_features, score as trained

                fx, guard = input_features(sql, x, obs)
                score = (
                    trained(fx, "search")
                    + 0.25 * x["prior"]
                    + self.cfg.get("guard", 1) * guard
                )
                self.verified[canonical(sql)] = (
                    trained(fx, "submit")
                    + 0.25 * x["prior"]
                    + self.cfg.get("guard", 1) * guard
                )
            if self.cfg.get("consistency"):
                from repair.verification import consistency

                delta = consistency(sql, obs, fx)
                score += delta
                self.verified[canonical(sql)] += delta
            if intent is not None:
                k = canonical(sql)
                hint_bonus = 0.5 * x["votes"]
                adjustment = (
                    self.cfg["intent"] * fidelity(parse(sql), intent) + hint_bonus
                )
                score += adjustment
                self.verified[k] = self.verified.get(k, score - adjustment) + adjustment
            if self.cfg.get("protect"):
                rank = legacy_order.get(canonical(sql))
                if rank is not None:
                    # Retain a vote from the independent warm-start ordering.
                    bonus = 2.2 / (1 + rank)
                    score += bonus
                    self.verified[canonical(sql)] += bonus
            k = emit(parse(sql))
            pool[k] = (score, sql)
            self.details[canonical(sql)] = score
        return pool

    def act(self, obs):
        if not self.cfg.get("policy"):
            return super().act(obs)
        history = obs["history"]
        mode = self.cfg.get("stop", "adaptive")
        if len(history) < 3:
            if len(history) == 1 and mode in ("adaptive", "intent"):
                return {"tool": "repair_hint"}
            return super().act(obs)
        if mode == "rescue" and not self.switched:
            if self.pool is None:
                self.pool = super().build(obs)
            original = super().act(obs)
            checked = [e for e in history if e["tool"] == "check"]
            if (
                any(e["observation"]["matches_example"] for e in checked)
                or obs["remaining"] > 3
            ):
                if original["tool"] != "submit" or any(
                    e["observation"]["matches_example"] for e in checked
                ):
                    return original
            self.switched = True
            self.pool = None
        if self.pool is None:
            self.pool = self.build(obs)
        if mode == "legacy":
            return super().act(obs)
        used = {canonical(e["sql"]): e for e in history if e["tool"] == "check"}
        verified = []
        for k, e in used.items():
            if e["observation"]["matches_example"]:
                v = self.details.get(k, -30)
                if self.cfg.get("judge"):
                    from repair.judge import feedback_features, score as trained

                    v = self.verified.get(k, v) + trained(
                        feedback_features(obs, e["observation"]), "submit"
                    )
                verified.append((v, e["sql"]))
        best = max(verified, default=None)
        pending = sorted(
            ((s, sql) for s, sql in self.pool.values() if canonical(sql) not in used),
            reverse=True,
        )
        if self.cfg.get("protect") and not best and len(used) > 1 and obs["remaining"]:
            from repair.verification import diversify

            failed = [
                e["sql"]
                for e in used.values()
                if not e["observation"]["matches_example"]
            ][-3:]
            pending.sort(key=lambda p: -(p[0] - diversify(p[1], failed)))
        if self.cfg.get("value_probe") and obs["remaining"]:
            from repair.verification import value_probe, apply_value

            if self.ground_query is not None and history[-1]["tool"] == "execute":
                new = apply_value(self.ground_query, history[-1]["observation"])
                if new and len(new.encode()) <= 16000 and canonical(new) not in used:
                    key = canonical(new)
                    parent = canonical(emit(self.ground_query[1]))
                    self.details[key] = self.details.get(parent, 0.0) + 2.0
                    self.verified[key] = self.verified.get(parent, 0.0) + 2.0
                    return {"tool": "check", "sql": new}
            sparse = obs["example"]["row_count"] == 0 or obs["example"]["rows"] in (
                [[0]],
                [[None]],
            )
            if (
                self.ground_query is None
                and obs["remaining"] >= 2
                and (not best or sparse)
                and (obs["remaining"] <= 3 or not pending)
            ):
                ranked = sorted(used, key=lambda k: -self.details.get(k, -30))
                self.ground_query = value_probe(
                    [used[k]["sql"] for k in ranked] + [s for _, s in pending[:3]], obs
                )
                if self.ground_query:
                    return {"tool": "execute", "sql": self.ground_query[0]}
        if self.cfg.get("judge") and best and pending:
            # Compare prospective vs verified candidates on the submission scale.
            # The public headers are unknown until checked. Spend a call in the
            # ambiguity band; stop once no unseen candidate is competitive.
            margin = self.cfg.get("margin", 1.2)
            competitive = [
                p
                for p in pending
                if self.verified.get(canonical(p[1]), p[0]) + margin >= best[0]
            ]
            pending = competitive
        if (
            obs["remaining"]
            and pending
            and (
                best is None or self.cfg.get("judge") or pending[0][0] > best[0] + 1e-6
            )
        ):
            # A candidate responsible for a previous binding error should not
            # consume the next slot unless no higher-quality alternative exists.
            if best is None and len(used) > 1:
                errors = [
                    e["observation"]["error"]["message"].lower().split(": ")[-1]
                    for e in used.values()
                    if e["observation"].get("error")
                    and "no such" in e["observation"]["error"]["message"].lower()
                ]
                pending.sort(
                    key=lambda z: (
                        -(z[0] - 2 * sum(term in z[1].lower() for term in set(errors)))
                    )
                )
            return {"tool": "check", "sql": pending[0][1]}
        if best:
            return {"tool": "submit", "sql": best[1]}
        return {
            "tool": "submit",
            "sql": max(
                (e["observation"]["ok"], self.details.get(k, -30), e["sql"])
                for k, e in used.items()
            )[2],
        }


def create_agent(checkpoint=None):
    return Agent(load_config(checkpoint or Path(__file__).with_name("config.json")))
