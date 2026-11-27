"""A small train-only index of structural properties, independent of identifiers."""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sqlglot import exp


def features(question):
    words = re.findall(r"[a-z]+|\d+", question.lower())
    words = ["#" if w.isdigit() else w for w in words]
    return set(words) | {" ".join(pair) for pair in zip(words, words[1:])}


def shape(tree):
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    flags = [
        bool(sel.args.get(x))
        for x in ("distinct", "group", "limit", "order", "having", "where")
    ]
    flags += [
        bool(tree.find(x))
        for x in (exp.Join, exp.Intersect, exp.Except, exp.Union, exp.Subquery)
    ]
    return sum(1 << i for i, x in enumerate(flags) if x)


@lru_cache(maxsize=1)
def memory():
    data = json.loads(Path(__file__).with_name("structure.json").read_text())
    vocab = {t: i for i, t in enumerate(data["vocab"])}
    df = Counter(t for row, _ in data["records"] for t in row)
    weight = {t: math.log1p(len(data["records"]) / (1 + n)) for t, n in df.items()}
    records = [
        (set(row), sum(weight[t] for t in row) ** 0.5, label)
        for row, label in data["records"]
    ]
    return vocab, weight, records


def estimate(question):
    vocab, w, rows = memory()
    query = {vocab[t] for t in features(question) if t in vocab}
    scores = [
        (sum(w[t] for t in sorted(query & row)) / max(norm, 1), -i, label)
        for i, (row, norm, label) in enumerate(rows)
    ]
    neighbors = sorted(scores, reverse=True)[:12]
    votes = [(s**3, label) for s, _, label in neighbors]
    total = sum(v for v, _ in votes)
    return [
        (sum(v for v, label in votes if label & (1 << i)) + 1) / (total + 2)
        for i in range(11)
    ]
