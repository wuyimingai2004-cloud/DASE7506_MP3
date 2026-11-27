"""Offline controller. All database information arrives through the episode history."""

import json
import re
from functools import lru_cache
from pathlib import Path

from sqlglot import exp

from repair.edits import emit, parse, main_select, projection_distance, rewrite_variants
from repair.structure import estimate, shape
from repair.relational import Catalog, relational_variants
from repair.semantics import semantic_variants
from repair.rank import signals, learned_score
from repair.grounding import anchor_probe, repair_from_probe


@lru_cache(maxsize=16)
def load_config(path):
    return json.loads(Path(path).read_text())


class Agent:
    def __init__(self, config):
        self.cfg = config
        self.pool = None
        self.attempts = {}
        self.probe = None

    def build(self, obs, collect=None):
        cfg = self.cfg
        q = obs["task"]["question"].lower()
        example = obs["example"]
        history = obs["history"]
        hints = next(e["observation"] for e in history if e["tool"] == "repair_hint")
        raw_schema = next(
            (e["observation"] for e in history if e["tool"] == "schema"), []
        )
        schema = {
            t["name"].lower(): {c["name"].lower() for c in t["columns"]}
            for t in raw_schema
        }
        catalog = Catalog(raw_schema)
        sources = [obs["issue_sql"]] + list(hints)[: cfg["hint_count"]]
        pool = {}
        seen = set()
        probs = estimate(q) if cfg["memory"] else []
        for pos, sql in enumerate(sources):
            tree = parse(sql)
            if tree is None:
                continue
            k = emit(tree)
            if k in seen:
                continue
            seen.add(k)
            variants = [(tree, "source")]
            variants.extend(rewrite_variants(tree, q, example, schema, cfg))
            if cfg.get("relational"):
                variants.extend(relational_variants(tree, q, example, catalog))
            if cfg.get("semantic"):
                variants.extend(semantic_variants(tree, q, example, catalog))
            for variant, kind in variants:
                candidate = sql if kind == "source" else emit(variant)
                identity = emit(variant)
                if len(candidate.encode()) > 16000:
                    continue
                sel = main_select(variant)
                score = (0.65 if pos else 0.0) - 0.035 * pos
                score += {
                    "source": 0,
                    "projection": -0.15,
                    "case": -1.2,
                    "group": -1.0,
                    "distinct": -1.2,
                    "bind": 0.9,
                    "prune": -0.9,
                    "group_key": -1.1,
                    "frequency": -0.3,
                    "set": 0.2,
                }.get(kind, -1.0)
                score -= (
                    2.4 * min(4, projection_distance(variant, example["columns"]))
                    if cfg["project"]
                    else 0
                )
                if cfg["local_edits"]:
                    wants_distinct = bool(
                        re.search(r"\b(distinct|different|unique)\b", q)
                    )
                    if wants_distinct and sel.args.get("distinct"):
                        score += 2.8
                if probs:
                    bits = shape(variant)
                    score += cfg["memory"] * sum(
                        (p - 0.5) * (1 if bits & (1 << i) else -1)
                        for i, p in enumerate(probs)
                    )
                if cfg.get("learned") or collect is not None:
                    values = signals(
                        variant, q, example["columns"], catalog, pos, kind, probs
                    )
                    if collect is not None:
                        collect.append((candidate, values))
                    if cfg.get("learned"):
                        score += cfg["learned"] * learned_score(values)
                if identity not in pool or score > pool[identity][0]:
                    pool[identity] = (score, candidate)
        return pool

    def act(self, obs):
        history = obs["history"]
        draft = obs["issue_sql"]
        if not history:
            return {"tool": "check", "sql": draft}
        first = history[0]["observation"]
        if len(history) == 1:
            sparse = obs["example"]["row_count"] == 0 or obs["example"]["rows"] in (
                [[0]],
                [[None]],
            )
            tree = parse(draft)
            mismatch = (
                tree is not None
                and projection_distance(tree, obs["example"]["columns"]) > 0
            )
            extra = self.cfg.get("semantic") and bool(
                re.search(
                    r"\b(distinct|different|both|most|least|never)\b",
                    obs["task"]["question"].lower(),
                )
            )
            risky = self.cfg["revisit"] and (
                sparse
                or mismatch
                or extra
                or (tree is not None and bool(tree.find(exp.Join)))
            )
            if first["matches_example"] and not risky:
                return {"tool": "submit", "sql": draft}
            return {"tool": "repair_hint"}
        if len(history) == 2 and self.cfg["schema"]:
            return {"tool": "schema"}
        if self.pool is None:
            self.pool = self.build(obs)
        last = history[-1]
        if not self.attempts:
            tree = parse(draft)
            self.attempts[emit(tree) if tree is not None else draft] = (draft, first)
        if last["tool"] == "check":
            checked_tree = parse(last["sql"])
            self.attempts[
                emit(checked_tree) if checked_tree is not None else last["sql"]
            ] = (last["sql"], last["observation"])
        best = max(
            (
                (self.pool.get(k, (-20, None))[0], sql)
                for k, (sql, res) in self.attempts.items()
                if res["matches_example"]
            ),
            default=None,
        )
        pending = sorted(
            (
                (score, sql)
                for k, (score, sql) in self.pool.items()
                if k not in self.attempts
            ),
            key=lambda it: -it[0],
        )
        if self.cfg.get("grounding") and best is None:
            if last["tool"] == "execute" and self.probe:
                _, tree, index, term = self.probe
                sql = repair_from_probe(tree, index, term, last["observation"])
                grounded_tree = parse(sql) if sql else None
                if (
                    grounded_tree is not None
                    and emit(grounded_tree) not in self.attempts
                    and obs["remaining"]
                ):
                    self.pool[emit(grounded_tree)] = (2.0, sql)
                    return {"tool": "check", "sql": sql}
            if (
                self.probe is None
                and obs["remaining"] >= 2
                and (not pending or obs["remaining"] == 2)
            ):
                self.probe = anchor_probe(self.attempts, obs)
                if self.probe:
                    return {"tool": "execute", "sql": self.probe[0]}
        if (
            pending
            and obs["remaining"]
            and (best is None or pending[0][0] > best[0] + 0.00001)
        ):
            return {"tool": "check", "sql": pending[0][1]}
        if best:
            return {"tool": "submit", "sql": best[1]}
        # A passing public query is preferred above. Otherwise favor a runnable, plausible query.
        fallback = max(
            (
                (res["ok"], self.pool.get(k, (-20, None))[0], sql)
                for k, (sql, res) in self.attempts.items()
            )
        )
        return {"tool": "submit", "sql": fallback[2]}


def create_agent(checkpoint=None):
    return Agent(load_config(checkpoint or Path(__file__).with_name("config.json")))
