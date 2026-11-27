"""Reference clause retrieval, candidate-relative evidence and tree inference, all offline."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import math
import re
from sqlglot import exp
from repair.edits import core, emit, parse, signature, tables, main_select
from repair.search import canonical
from repair.preferences import context, features, properties, stem
from transforms import tags, sources


@lru_cache(maxsize=4096)
def clauses(sql):
    t = parse(sql)
    if t is None:
        return {}
    s = main_select(t)

    def sketch(n, depth):
        if n is None:
            return "_"
        if isinstance(n, exp.Alias):
            return sketch(n.this, depth)
        if isinstance(n, exp.Column):
            return "col"
        if isinstance(n, exp.Literal):
            return "str" if n.is_string else "num"
        if depth <= 0:
            return n.key
        kids = [sketch(c, depth - 1) for c in n.iter_expressions()]
        return n.key + ("(" + ",".join(kids) + ")" if kids else "")

    x = {
        k: sketch(s.args.get(k), 3)
        for k in ("where", "order", "having", "group", "limit", "distinct")
    }
    x["project"] = ",".join(sketch(n, 2) for n in s.expressions)
    x["root"] = t.key
    x["width"] = str(len(tables(s)))
    x["subqueries"] = str(len(list(t.find_all(exp.Select))))
    x["direction"] = str([bool(n.args.get("desc")) for n in s.find_all(exp.Ordered)])
    return x


def question_keys(o):
    words = tags(o, o["history"][2]["observation"])
    hs = sources(o)
    shapes = [clauses(s) for s in hs]
    paths = {"D/" + k + "/" + v for k, v in shapes[0].items()}
    paths.update("H/" + k + "/" + v for s in shapes[1:] for k, v in s.items())
    return words, paths


def get_neighbors(keys, records):
    a, b = keys

    def similarity(c, d):
        return len(c & d) / max(1, math.sqrt(len(c) * len(d)))

    ranked = sorted(
        (
            (0.66 * similarity(a, c) + 0.34 * similarity(b, d), i, g)
            for i, (c, d, g) in enumerate(records)
        ),
        key=lambda z: (-z[0], z[1]),
    )[:20]
    sums = {}
    total = sum(s**7 for s, _, _ in ranked) or 1
    for s, _, g in ranked:
        for k, v in g.items():
            sums.setdefault(k, Counter())[v] += s**7 / total
    return sums


@lru_cache(maxsize=1)
def load_policy():
    return json.loads(Path(__file__).with_name("search_model.json").read_text())


@lru_cache(maxsize=1)
def ref_library():
    return [(set(w), set(p), c) for w, p, c in load_policy()["grammar"]]


def reference_context(o, lib):
    n = get_neighbors(question_keys(o), lib)
    schema = {r["name"].lower(): r for r in o["history"][2]["observation"]}
    return n, schema


def additional(sql, o, ctx):
    neigh, schema = ctx
    t = parse(sql)
    sel = main_select(t)
    shape = clauses(sql)
    x = {"ref/" + k: probs.get(shape.get(k), 0) for k, probs in neigh.items()}
    x["ref/max_mismatch"] = sum(v < 0.15 for v in x.values())
    x["ref/mean"] = sum(x.get("ref/" + k, 0) for k in shape) / max(1, len(shape))
    valid = tables(sel)
    bad_owner = 0
    case_conflict = 0
    type_conflict = 0
    ambiguous = 0
    headers = o["example"]["columns"]
    for i, e in enumerate(sel.expressions):
        if i >= len(headers):
            break
        col = core(e)
        if not isinstance(col, exp.Column):
            continue
        owners = [
            s for a, s in valid.items() if not col.table or a == col.table.lower()
        ]
        matches = [
            c
            for s in owners
            for c in schema.get(s, {}).get("columns", ())
            if c["name"].lower() == col.name.lower()
        ]
        ambiguous += len(matches) > 1
        bad_owner += not matches
        if not matches:
            continue
        case_conflict += all(c["name"] != headers[i] for c in matches)
        values = [r[i] for r in o["example"]["rows"] if len(r) > i and r[i] is not None]
        number = any(isinstance(v, (int, float)) for v in values)
        string = any(isinstance(v, str) for v in values)
        types = [c["type"].lower() for c in matches]
        type_conflict += (
            number
            and all(re.search("char|text|clob", z) for z in types)
            or string
            and all(re.search("int|real|float|double", z) for z in types)
        )
    x.update(
        bad_output_owner=bad_owner,
        output_case_conflict=case_conflict,
        output_type_conflict=type_conflict,
        ambiguous_output=ambiguous,
    )
    names = {n.name.lower() for n in t.find_all(exp.Table)}
    mentioned = stem(o["task"]["question"])
    x["unbound_relations"] = len(names - set(schema))
    duplicated = len({tuple(r) for r in o["example"]["rows"]}) < len(
        o["example"]["rows"]
    )
    x["distinct_impossible"] = int(
        duplicated and bool(sel.args.get("distinct")) and isinstance(t, exp.Select)
    )
    x["sparse"] = float(o["example"]["rows"] in ([], [[None]], [[0]]))
    order = sel.args.get("order")
    x["ordered_status"] = 1 if bool(order) == o["ordered"] else -1
    literals = list(t.find_all(exp.Literal))
    q = o["task"]["question"]
    x["case_sensitive_quotes"] = sum(l.is_string and l.this in q for l in literals)
    x["misspelled_quotes"] = sum(
        l.is_string and l.this.casefold() not in q.casefold() for l in literals
    )
    x["qnumber_unused"] = sum(
        v not in {l.this for l in literals} for v in re.findall(r"\b\d+\b", q)
    )
    x["projection_duplicates"] = len(sel.expressions) - len(
        {emit(core(p)) for p in sel.expressions}
    )
    prefix = re.split(r"\b(?:where|whose|having|with|that|who)\b", q, flags=re.I)[0]
    for slot, txt in [("prefix", prefix), ("all", q)]:
        words = stem(txt)
        x["table_" + slot] = sum(
            len(stem(n) & words) / max(1, len(stem(n))) for n in names
        )
    return x


def forest_score(x, forest):
    total = 0
    for tree in forest:
        node = tree
        while isinstance(node, list):
            key, threshold, left, right = node
            node = left if x.get(key, 0) <= threshold else right
        total += node
    return total / len(forest)


def expand_features(raw, sql, o, ref):
    x = {k: v for k, v in raw.items() if not k.startswith("cue/")}
    x.update(additional(sql, o, ref))
    return x
