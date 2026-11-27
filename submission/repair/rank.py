"""Small train-only linear reranker over schema-independent features."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from sqlglot import exp

from repair.edits import main_select, projection_distance, tables
from repair.relational import alignment, name_words

TAGS = (
    "source",
    "projection",
    "distinct",
    "case",
    "group",
    "bind",
    "prune",
    "group_key",
    "frequency",
    "set",
    "inner",
    "count",
    "group_sum",
    "anti",
    "extreme",
    "name_filter",
)


def signals(tree, q, headers, cat, pos, tag, probs):
    sel = main_select(tree)
    owners = tables(sel)
    names = [t.name for t in tree.find_all(exp.Table)]
    cols = [c.name for c in tree.find_all(exp.Column)]
    content = {w for s in names + cols for w in name_words(s) if len(w) > 1}
    overlap = sum(alignment(w, q) > 0 for w in sorted(content))
    projected = [p for p in sel.expressions if p.find(exp.Column)]
    joins = sel.args.get("joins", [])
    foreign = bad = 0
    for j in joins:
        on = j.args.get("on")
        if (
            not isinstance(on, exp.EQ)
            or not isinstance(on.this, exp.Column)
            or not isinstance(on.expression, exp.Column)
        ):
            continue
        a, b = on.this, on.expression
        src, dst = owners.get(a.table.lower()), owners.get(b.table.lower())
        if src and dst:
            if (dst, a.name.lower(), b.name.lower()) in cat.graph[src]:
                foreign += 1
            elif a.name.lower() != b.name.lower():
                bad += 1
    invalid = sum(
        c.table.lower() in owners
        and c.name.lower() not in cat.columns.get(owners[c.table.lower()], set())
        for c in tree.find_all(exp.Column)
    )
    literal = [str(x.this).lower() for x in tree.find_all(exp.Literal) if x.is_string]
    clean = [s.strip("%") for s in literal]
    mentioned = sum(bool(x) and x in q for x in clean)
    numeric = [str(x.this) for x in tree.find_all(exp.Literal) if not x.is_string]
    bits = 0
    from repair.structure import shape

    bits = shape(tree)
    cues = [
        (r"\b(distinct|unique|different)\b", 0),
        (r"\b(each|per|every|at least)\b", 1),
        (r"\b(most|highest|lowest|least|largest|smallest|fewest)\b", 2),
        (r"\b(order|ordered|sorted|descending|ascending)\b", 3),
        (r"\b(both)\b", 7),
        (r"\b(no|never|not|except|without)\b", 8),
    ]
    cue_vals = [
        int(bool(re.search(regex, q))) * (1 if bits & (1 << i) else -1)
        for regex, i in cues
    ]
    np = max(1, len(names))
    data = {
        "draft": int(pos == 0),
        "hint_position": min(pos, 4) / 4,
        "header_distance": min(projection_distance(tree, headers), 4),
        "tables": min(6, len(names)),
        "joins": min(5, len(joins)),
        "table_alignment": sum(alignment(t, q) > 0 for t in names) / np,
        "unmentioned_tables": sum(alignment(t, q) == 0 for t in names),
        "words_overlap": min(10, overlap),
        "fraction_overlap": overlap / max(1, len(content)),
        "projection_alignment": sum(alignment(p.sql(), q) > 0 for p in projected)
        / max(1, len(projected)),
        "valid_fk": min(4, foreign),
        "dubious_join": min(3, bad),
        "invalid_qualifier": min(4, invalid),
        "mentioned_literals": mentioned,
        "ungrounded_literals": len(clean) - mentioned,
        "matched_numbers": sum(n in q.split() for n in numeric),
        "left": sum(j.side == "LEFT" for j in joins),
        "size": math.log1p(sum(1 for _ in tree.walk())),
        "predicates": min(5, len(list(tree.find_all(exp.Predicate)))),
        "casefold": int(bool(tree.find(exp.Lower))),
        "count_distinct": sum(
            isinstance(c.this, exp.Distinct) for c in tree.find_all(exp.Count)
        ),
    }
    data.update({f"cue_{i}": v for (_, i), v in zip(cues, cue_vals)})
    for i, p in enumerate(probs):
        data[f"memory_{i}"] = (2 * p - 1) * (1 if bits & (1 << i) else -1)
        data[f"bit_{i}"] = int(bool(bits & (1 << i)))
    data.update({f"edit_{t}": int(tag == t) for t in TAGS})
    return data


@lru_cache(maxsize=1)
def weights():
    return json.loads(Path(__file__).with_name("rank.json").read_text())


def learned_score(features):
    w = weights()
    return sum(w.get(k, 0.0) * v for k, v in features.items())
