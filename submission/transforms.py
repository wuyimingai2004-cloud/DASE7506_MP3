"""Transfer small, schema-independent clause edits from train references to new sources."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json
import math
import re
from sqlglot import exp
from repair.edits import core, emit, parse, tables
from repair.sketch import sketch_tokens
from repair.relational import Catalog

SLOTS = ("expressions", "where", "group", "having", "order", "limit", "distinct")


def sequence(tree, clause):
    value = tree.args.get(clause)
    return value if isinstance(value, list) else [value] if value is not None else []


def encode(nodes, bindings, learn):
    """Typed positional variables. Never store an identifier, free string or task literal."""

    def visit(n):
        if isinstance(n, exp.Column):
            if n.is_star:
                return ["Star", {}]
            key = ("column", n.name.lower())
        elif isinstance(n, exp.Literal):
            key = ("text" if n.is_string else "num", n.this)
            if not n.is_string and n.this in ("0", "1"):
                return ["constant", n.this]
        elif isinstance(n, exp.Star):
            return ["Star", {}]
        elif isinstance(n, exp.Expression):
            if isinstance(
                n,
                (
                    exp.Identifier,
                    exp.Table,
                    exp.Select,
                    exp.Subquery,
                    exp.Anonymous,
                    exp.Var,
                ),
            ):
                raise ValueError
            if isinstance(n, exp.Alias):
                return visit(n.this)
            return [
                type(n).__name__,
                {k: visit(v) for k, v in sorted(n.args.items()) if v is not None},
            ]
        elif isinstance(n, list):
            return [visit(v) for v in n]
        elif type(n) is bool:
            return n
        else:
            raise ValueError
        if key not in bindings:
            if not learn:
                raise ValueError
            bindings[key] = n
        return ["slot", list(bindings).index(key)]

    return [visit(n) for n in nodes]


def decode(tree, bindings):
    if isinstance(tree, bool):
        return tree
    if len(tree) == 2 and isinstance(tree[0], str):
        kind, args = tree
        if kind == "slot":
            return bindings[args].copy()
        if kind == "constant":
            return exp.Literal.number(args)
        return getattr(exp, kind)(**{k: decode(v, bindings) for k, v in args.items()})
    return [decode(v, bindings) for v in tree]


def signature(tree, clause):
    bindings = {}
    before = encode(sequence(tree, clause), bindings, True)
    # Include selected columns and grouping/order/literal handles defined outside the edit.
    # Use a stable traversal order. Only columns/literals visible in this scope can bind.
    for part in ("expressions", "group", "order", "where", "having"):
        for item in sequence(tree, part):
            for n in item.walk():
                if isinstance(n, (exp.Column, exp.Literal)):
                    encode([n], bindings, True)
    return json.dumps(before, sort_keys=True, separators=(",", ":")), bindings


def tags(obs, schema):
    return sketch_tokens(
        obs["task"]["question"], obs["example"]["columns"], Catalog(schema)
    ) | {
        "order=" + str(obs["ordered"]),
        "size=" + str(min(2, obs["example"]["row_count"])),
    }


def sources(obs):
    return list(dict.fromkeys([obs["issue_sql"], *obs["history"][1]["observation"]]))


def eligible(tree):
    return (
        isinstance(tree, exp.Select)
        and not tree.find(exp.Subquery)
        and not tree.args.get("with")
    )


def mine(rows, obs, refs):
    examples = defaultdict(list)
    for row, o in zip(rows, obs):
        assert row["id"] == o["task"]["id"]
        gold = parse(refs[row["id"]])
        if not eligible(gold):
            continue
        context = sorted(tags(o, o["history"][2]["observation"]))
        seen = set()
        for s in sources(o):
            t = parse(s)
            if not eligible(t):
                continue
            for part in SLOTS:
                try:
                    key, b = signature(t, part)
                    after = encode(sequence(gold, part), b, False)
                except ValueError:
                    continue
                value = json.dumps(after, sort_keys=True, separators=(",", ":"))
                if value == key or (part, key, value) in seen:
                    continue
                seen.add((part, key, value))
                examples[part + "|" + key].append([context, value])
    return examples


@lru_cache(maxsize=1)
def library():
    d = json.loads(Path(__file__).with_name("transforms.json").read_text())
    return {k: [(frozenset(t), v) for t, v in a] for k, a in d.items()}


def edits(t, context):
    if not eligible(t):
        return
    actions = []
    for part in SLOTS:
        try:
            key, b = signature(t, part)
        except ValueError:
            continue
        near = []
        for other, value in library().get(part + "|" + key, ()):
            sim = len(other & context) / max(1, math.sqrt(len(other) * len(context)))
            near.append((sim, value))
        votes = Counter()
        for sim, value in sorted(near, reverse=True)[:8]:
            votes[value] += sim**4
        for value, weight in votes.most_common(2):
            if weight < 0.09:
                continue
            try:
                replacement = decode(json.loads(value), list(b.values()))
            except IndexError:
                continue
            if not replacement and part == "expressions":
                continue
            v = t.copy()
            v.set(
                part,
                replacement
                if part == "expressions"
                else replacement[0]
                if replacement
                else None,
            )
            actions.append((weight, part, v))
    for w, p, v in sorted(actions, key=lambda a: -a[0])[:6]:
        yield v, "transfer_" + p, min(1.5, w)
