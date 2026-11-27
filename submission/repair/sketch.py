"""Retrieve identifier-free SQL sketches and instantiate them against the observed schema."""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sqlglot import exp
from repair.edits import core, emit, header, main_select, tables
from repair.relational import name_words

STOP = set(
    "what which show give me list find return tell are is the a an of all in that who and for to with on their do does there has have display please".split()
)
AGG = {
    "count": exp.Count,
    "sum": exp.Sum,
    "max": exp.Max,
    "min": exp.Min,
    "avg": exp.Avg,
}
COMPARISON = {"gt": exp.GT, "gte": exp.GTE, "eq": exp.EQ, "lt": exp.LT, "lte": exp.LTE}


def stem(w):
    return (
        w[:-3] + "y"
        if len(w) > 4 and w.endswith("ies")
        else w.rstrip("s")
        if len(w) > 3
        else w
    )


def words(s):
    return {stem(w) for w in name_words(s) if w not in STOP}


def sketch_tokens(question, headers, cat):
    schema_words = {
        w for t, cols in cat.columns.items() for name in (t, *cols) for w in words(name)
    }
    seq = []
    for w in re.findall(r"[a-z]+|\d+", question.lower()):
        v = stem(w)
        if w in STOP:
            continue
        seq.append("#" if w.isdigit() else "_" if v in schema_words else v)
    tokens = set(seq) | {a + " " + b for a, b in zip(seq, seq[1:])}
    tokens.update(
        "proj_" + (h.key if h is not None else "?") for h in map(header, headers)
    )
    tokens.add("width_" + str(min(4, len(headers))))
    return tokens


def program(tree):
    if (
        not isinstance(tree, exp.Select)
        or tree.find(exp.Subquery)
        or tree.args.get("with")
    ):
        return None
    src = tables(tree)
    if not 1 <= len(src) <= 4 or len(set(src.values())) != len(src):
        return None
    group = tree.args.get("group")
    order = tree.args.get("order")
    having = tree.args.get("having")
    sort = "none"
    desc = False
    if order:
        if len(order.expressions) != 1:
            return None
        expr = order.expressions[0]
        desc = bool(expr.args.get("desc"))
        sort = expr.this.key
        if sort not in {*AGG, "column"}:
            return None
    cap = tree.args.get("limit")
    if cap and cap.expression.name != "1":
        return None
    cond = having.this if having else None
    if having and (
        cond.key not in COMPARISON
        or not isinstance(cond.this, exp.Count)
        or not isinstance(cond.expression, exp.Literal)
    ):
        return None
    gr = "none"
    if group:
        p = {
            c.name.lower()
            for x in tree.expressions
            if isinstance(core(x), exp.Column)
            for c in core(x).find_all(exp.Column)
        }
        gr = (
            "display"
            if all(
                isinstance(c, exp.Column) and c.name.lower() in p
                for c in group.expressions
            )
            else "key"
        )
    return (
        len(src),
        bool(tree.args.get("distinct")),
        bool(tree.args.get("where")),
        gr,
        sort,
        desc,
        bool(cap),
        cond.key if cond is not None else "",
    )


@lru_cache(maxsize=1)
def library():
    data = json.loads(Path(__file__).with_name("templates.json").read_text())
    return [(set(t), tuple(p)) for t, p in data]


def retrieve(tokens):
    votes = Counter()
    near = []
    for i, (other, p) in enumerate(library()):
        score = len(tokens & other) / max(1, math.sqrt(len(tokens) * len(other)))
        near.append((score, -i, p))
    for sim, _, p in sorted(near, reverse=True)[:16]:
        votes[p] += sim**4
    total = sum(votes.values()) or 1
    return [(p, v / total) for p, v in votes.most_common(5)]


def owners(col, old, cat):
    name = col.name.lower()
    preferred = old.get(col.table.lower(), col.table.lower())
    opts = [t for t, columns in cat.columns.items() if name in columns]
    return sorted(opts, key=lambda t: (t != preferred, t not in old.values(), t))


def qualify(expr, old, cat, preferred):
    out = expr.copy()
    needed = []
    for c in out.find_all(exp.Column):
        opts = owners(c, old, cat)
        if not opts:
            return None, []
        old_owner = old.get(c.table.lower())
        t = next((t for t in [old_owner, *preferred] if t in opts), opts[0])
        c.set("table", exp.to_identifier(t, quoted=True))
        c.set("this", exp.to_identifier(c.name.lower(), quoted=True))
        needed.append(t)
    return out, needed


def from_graph(sel, required, cat):
    root = required[0]
    result = sel.from_(exp.to_table(root, quoted=True))
    joined = {root: None}
    for t in required[1:]:
        if t in joined:
            continue
        path = cat.path(joined, t)
        if path is None or len(path) > 3:
            return None, []
        for a, x, b, y in path:
            result = result.join(
                exp.to_table(b, quoted=True),
                on=exp.column(x, table=a, quoted=True).eq(
                    exp.column(y, table=b, quoted=True)
                ),
                copy=False,
            )
            joined[b] = None
    return result, list(joined)


def templates(tree, q, example, cat, neighbors):
    if not isinstance(tree, exp.Select) or not cat.columns:
        return
    old = tables(tree)
    target = list(map(header, example["columns"]))
    if not target or any(x is None for x in target):
        return
    qwords = words(q)
    ranked = sorted(
        cat.columns, key=lambda t: (-len(qwords & words(t)), t not in old.values(), t)
    )
    preferred = [t for t in ranked if t in old.values()]
    project = []
    required = []
    for i, h in enumerate(target):
        # Preserve explicit ownership when the original/hint names the output column.
        ref = next(
            (
                c
                for c in (
                    tree.expressions[i].find_all(exp.Column)
                    if i < len(tree.expressions)
                    else []
                )
                if any(a.name.lower() == c.name.lower() for a in h.find_all(exp.Column))
            ),
            None,
        )
        if ref is not None and ref.table:
            for c in h.find_all(exp.Column):
                if c.name.lower() == ref.name.lower():
                    c.set("table", ref.args["table"].copy())
        obj, ts = qualify(h, old, cat, required + preferred)
        if obj is None:
            return
        project.append(obj)
        required += ts
    raw_where = tree.args.get("where")
    wh, wt = (
        qualify(raw_where, old, cat, required + preferred)
        if raw_where and not raw_where.find(exp.Select)
        else (None, [])
    )
    order = tree.args.get("order")
    ordercol = None
    if order and order.expressions:
        ordercol = order.expressions[0].this.find(exp.Column)
    displayed = [p for p in project if isinstance(p, exp.Column)]
    nums = re.findall(r"\b\d+\b", q)
    for pattern, support in neighbors:
        width, unique, filters, group, sort, desc, cap, having = pattern
        if filters and wh is None or having and not nums:
            continue
        use = required + (wt if filters else [])
        expr = None
        if sort == "count":
            expr = exp.Count(this=exp.Star())
        elif sort != "none":
            raw = (
                ordercol
                if ordercol is not None
                else (displayed[0] if displayed else None)
            )
            if raw is None:
                continue
            expr, ts = qualify(raw, old, cat, use + preferred)
            if expr is None:
                continue
            use = use + ts
            if sort != "column":
                expr = AGG[sort](this=expr)
        start = list(dict.fromkeys(use))
        extras = list(dict.fromkeys([*preferred, *ranked]))
        if not start:
            start = [extras[0]]
        choices = [start]
        if width > len(start):
            extras = [
                t
                for t in extras
                if t not in start and (t in preferred or words(t) & qwords)
            ]
            choices = [start + [t] for t in extras[:3]]
        for choice in choices:
            sel, joined = from_graph(
                exp.select(*(p.copy() for p in project)), choice, cat
            )
            if sel is None or len(joined) != width:
                continue
            if filters:
                sel.set("where", wh.copy())
            if unique:
                sel.set("distinct", exp.Distinct())
            if group != "none":
                keys = displayed
                if group == "key":
                    table = displayed[0].table if displayed else joined[0]
                    keys = [
                        exp.column(k, table=table, quoted=True)
                        for k in sorted(cat.pk.get(table, ()))
                    ]
                if not keys:
                    continue
                sel = sel.group_by(*(k.copy() for k in keys))
            if having:
                sel = sel.having(
                    COMPARISON[having](
                        this=exp.Count(this=exp.Star()),
                        expression=exp.Literal.number(nums[-1]),
                    )
                )
            if expr is not None:
                sel = sel.order_by(exp.Ordered(this=expr.copy(), desc=desc))
            if cap:
                sel = sel.limit(1)
            yield sel, support
