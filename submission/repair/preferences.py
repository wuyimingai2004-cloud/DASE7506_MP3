"""Question-conditioned SQL features shared by the two listwise decision heads."""

import math
import re
from functools import lru_cache
from pathlib import Path
import json
from sqlglot import exp
from repair.edits import core, main_select, parse, tables
from repair.judge import input_features, feedback_features
from repair.intent import attributes
from repair.verification import consistency
from repair.search import canonical
from repair.sketch import words

CUES = {
    "all": r"\b(all|every|each)\b",
    "how_many": r"\b(number|count|how many)\b",
    "superlative": r"\b(most|least|fewest|largest|smallest|youngest|oldest|highest|lowest|longest|shortest)\b",
    "group": r"\b(for each|each|per|every)\b",
    "unique": r"\b(distinct|different|unique)\b",
    "plural": r"\b(names|ids|codes|types)\b",
    "sort": r"\b(sort\w*|order\w*|alphabetic\w*)\b",
    "neg": r"\b(not|never|except|without|no|neither)\b",
    "both": r"\bboth\b",
    "or": r"\b(or|either)\b",
    "threshold": r"\b(at least|at most|less than|more than|greater than)\b",
    "average": r"\b(average|mean)\b",
    "total": r"\b(total|sum)\b",
}
IGNORE = {
    "a",
    "an",
    "of",
    "the",
    "id",
    "code",
    "name",
    "number",
    "and",
    "in",
    "to",
    "by",
}


@lru_cache(maxsize=2)
def weights():
    return json.loads(Path(__file__).with_name("preferences.json").read_text())


def stem(text):
    return {
        v[:-3] + "y"
        if len(v) > 4 and v.endswith("ies")
        else v[:-1]
        if len(v) > 3 and v.endswith("s")
        else v
        for v in re.findall(
            r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
        )
    }


def context(o):
    q = o["task"]["question"].lower()
    cues = {k for k, r in CUES.items() if re.search(r, q)}
    if o["ordered"]:
        cues.add("ordered")
    if o["example"]["row_count"] == 0 or o["example"]["rows"] in ([[0]], [[None]]):
        cues.add("sparse")
    ds = o["history"][0]["observation"]
    if ds["matches_example"]:
        cues.add("draft_pass")
    hints = next(e["observation"] for e in o["history"] if e["tool"] == "repair_hint")
    sources = [o["issue_sql"]] + hints
    schema = next(e["observation"] for e in o["history"] if e["tool"] == "schema")
    return (
        q,
        cues,
        stem(q),
        {t["name"].lower(): stem(t["name"]) - IGNORE for t in schema},
        [properties(s) for s in sources],
    )


@lru_cache(maxsize=2048)
def properties(sql):
    t = parse(sql)
    if t is None:
        return {}, {}
    a = attributes(t)
    s = main_select(t)

    def names(node, cls):
        return (
            frozenset(c.name.lower() for c in node.find_all(cls))
            if node is not None
            else frozenset()
        )

    order = s.args.get("order")
    a.update(
        direction="desc"
        if order and order.expressions[0].args.get("desc")
        else "asc"
        if order
        else "none",
        agg="/".join(
            sorted({c.key for e in s.expressions for c in e.find_all(exp.AggFunc)})
        )
        or "none",
    )
    roles = {
        "table": names(t, exp.Table),
        "projection": frozenset(n for p in s.expressions for n in names(p, exp.Column)),
        "literal": frozenset(v.this.lower() for v in t.find_all(exp.Literal)),
    }
    for k in ("where", "order", "group", "having"):
        roles[k] = names(s.args.get(k), exp.Column)
    return a, roles


def features(sql, raw, o, ctx):
    q, cues, qw, entities, sources = ctx
    t = parse(sql)
    x, guard = input_features(sql, raw, o)
    x["evidence_guard"] = guard
    x["consistency"] = consistency(sql, o, x)
    x = {k: max(-4, min(4, v)) for k, v in x.items() if v}
    a, roles = properties(sql)
    has = roles["table"]
    mentioned = {t for t, w in entities.items() if w and w <= qw}
    x["matched_entities"] = min(4, len(has & mentioned))
    x["missing_entities"] = min(4, len(mentioned - has))
    x["no_entity_match"] = int(bool(mentioned) and not has & mentioned)
    for k, v in a.items():
        x[f"attr/{k}/{v}"] = 1
        x["draft_agrees/" + k] = int(sources[0][0].get(k) == v)
        x["hints_agree/" + k] = sum(z.get(k) == v for z, _ in sources[1:]) / max(
            1, len(sources) - 1
        )
        for cue in sorted(cues):
            x[f"cue/{cue}/{k}/{v}"] = 1
    for k, v in roles.items():
        x["match_draft/" + k] = int(v == sources[0][1].get(k))
        x["match_hints/" + k] = sum(v == r.get(k) for _, r in sources[1:]) / max(
            1, len(sources) - 1
        )
        if k not in ("table", "literal"):
            x["mention_role/" + k] = min(
                3, sum(bool((stem(n) - IGNORE) & qw) for n in v)
            )
    # Local syntactic context independent of identifiers and literal values.
    ignore = {"identifier", "tablealias", "alias", "column", "table", "from"}
    for n in t.walk():
        if n.key in ignore or n.parent is None:
            continue
        k = f"path/{n.parent.key}/{n.arg_key}/{n.key}"
        x[k] = min(3, x.get(k, 0) + 1)
    return {k: v for k, v in x.items() if v}


def feedback(o, fb):
    return feedback_features(o, fb)


def score(x, w):
    return sum(v * w.get(k, 0.0) for k, v in x.items())


def family(sql):
    """A failure of membership cannot be fixed by sorting the very same rows on an unordered task."""
    t = parse(canonical(sql))
    if (
        not isinstance(t, exp.Select)
        or t.args.get("limit")
        or t.args.get("offset")
        or t.find(exp.Window)
    ):
        return None
    t = t.copy()
    t.set("order", None)
    t.set("distinct", None)
    return t.sql(dialect="sqlite")
