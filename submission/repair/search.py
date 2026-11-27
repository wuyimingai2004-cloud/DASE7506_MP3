"""Additional bounded candidates and schema-independent pairwise ranking features."""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sqlglot import exp
from repair.edits import (
    core,
    emit,
    main_select,
    parse,
    projection,
    projection_distance,
    tables,
)
from repair.sketch import from_graph, program, qualify, words


def relink(tree, cat, q):
    """Rebuild the smallest declared join graph connecting clause-referenced tables."""
    old = tables(tree)
    if (
        not old
        or len(old) != len(set(old.values()))
        or tree.find(exp.Subquery)
        or tree.args.get("with")
    ):
        return
    if any(
        j.side in ("LEFT", "RIGHT", "FULL") for j in tree.args.get("joins", ())
    ) and re.search(r"\b(no|not|without|never)\b", q):
        return
    use = []
    clean = tree.copy()
    clean.set("joins", None)
    clean.set("from", None)
    for c in clean.find_all(exp.Column):
        fixed, needed = qualify(c, old, cat, use + list(old.values()))
        if fixed is None:
            return
        c.replace(fixed)
        use += needed
    use = list(dict.fromkeys(use))
    if not use:
        return
    out, joined = from_graph(clean, use, cat)
    if out is not None:
        yield out, "minimal_fk"
        # Count and group operations can depend on a relation which supplies no
        # projected attribute. Retain mentioned source relations in a second path.
        keep = [
            t
            for t in old.values()
            if t in cat.columns and t not in joined and words(t) & words(q)
        ]
        if keep:
            other, _ = from_graph(clean, use + keep, cat)
            if other is not None:
                yield other, "mentioned_fk"


def edits(tree, q, example, cat):
    if not isinstance(tree, exp.Select):
        return
    base = projection(tree, example["columns"], cat.columns) or tree
    yield from relink(base, cat, q)
    if base.args.get("group"):
        for c in base.find_all(exp.Count):
            if not isinstance(c.this, exp.Star):
                out = base.copy()
                for count in out.find_all(exp.Count):
                    count.set("this", exp.Star())
                yield from relink(out, cat, q)
                break
        if base.args.get("order") and base.args.get("limit"):
            out = base.copy()
            out.args["order"].expressions[0].set("this", exp.Count(this=exp.Star()))
            yield out, "count_order"
            yield from relink(out, cat, q)
    wh = base.args.get("where")
    if not wh:
        return
    if isinstance(wh.this, exp.And):
        for pred in (wh.this.this, wh.this.expression):
            others = base.where(pred.copy(), append=False)
            yield others, "one_filter"
            yield from relink(others, cat, q)
    if any(isinstance(p, exp.Literal) and p.is_string for p in wh.walk()):
        for form in (str.lower, str.upper, str.title, str.strip):
            out = base.copy()
            changed = False
            for lit in out.args["where"].find_all(exp.Literal):
                if lit.is_string and form(lit.this) != lit.this:
                    lit.set("this", form(lit.this))
                    changed = True
            if changed:
                yield out, "literal_form"
                yield from relink(out, cat, q)
    # Set difference over the projected entity and its participating records.
    if re.search(r"\b(no|not|never|without|except|neither)\b", q) and base.args.get(
        "joins"
    ):
        left = base.copy()
        for key in ("where", "order", "limit", "distinct"):
            left.set(key, None)
        right = left.copy()
        cond = wh.this
        if isinstance(cond, (exp.NEQ, exp.EQ)) and isinstance(
            cond.expression, exp.Literal
        ):
            right = right.where(cond.this.copy().eq(cond.expression.copy()))
        elif not cond.find(exp.Null):
            return
        for j in right.args.get("joins", []):
            j.set("side", None)
        for small, _ in relink(left, cat, ""):
            if len(tables(small)) < len(tables(right)):
                yield small.except_(right), "anti_join"
                break


@lru_cache(maxsize=4096)
def canonical(sql):
    """Merge simple aliases only when all name bindings remain unambiguous."""
    t = parse(sql)
    if not isinstance(t, exp.Select) or t.find(exp.Subquery) or t.args.get("with"):
        return emit(t) if t is not None else sql
    old = tables(t)
    if len(set(old.values())) != len(old) or not old:
        return emit(t)
    out = t.copy()
    if any(isinstance(z, exp.Alias) for z in out.expressions):
        aliases = {z.alias.lower() for z in out.expressions if z.alias}
        if not any(
            c.name.lower() in aliases
            for k in ("where", "group", "having", "order", "joins")
            for item in (
                out.args.get(k)
                if isinstance(out.args.get(k), list)
                else [out.args.get(k)]
            )
            if item
            for c in item.find_all(exp.Column)
        ):
            out.set("expressions", [core(z).copy() for z in out.expressions])
    for t in out.find_all(exp.Table):
        t.set("alias", None)
        t.set("this", exp.to_identifier(t.name.lower(), quoted=True))
    for c in out.find_all(exp.Column):
        # SQLite may interpret an unknown double-quoted name as a string.
        if c.this.args.get("quoted") and not c.table:
            continue
        if c.table.lower() in old:
            c.set("table", exp.to_identifier(old[c.table.lower()], quoted=True))
        elif not c.table and len(old) == 1:
            c.set("table", exp.to_identifier(next(iter(old.values())), quoted=True))
        if not c.is_star:
            c.set("this", exp.to_identifier(c.name.lower(), quoted=True))
    return emit(out)


def values(tree, q, example, cat, neighbors):
    sel = main_select(tree)
    src = tables(sel)
    joins = sel.args.get("joins", [])
    schema_names = set(src.values())
    refs = Counter()
    qs = words(q)
    cols = set()
    invalid = 0
    for c in tree.find_all(exp.Column):
        cn = c.name.lower()
        t = src.get(c.table.lower())
        if c.find_ancestor(exp.Join) is None:
            cols.update(words(cn))
            if t:
                refs[t] += 1
            else:
                for owner in schema_names:
                    if cn in cat.columns.get(owner, ()):
                        refs[owner] += 1
        if t and cn not in cat.columns.get(t, ()) and not c.is_star:
            invalid += 1
    table_words = {w for name in schema_names for w in words(name)}
    pattern = program(tree)

    def prob(f):
        return sum(v for p, v in neighbors if f(p))

    grp = sel.args.get("group")
    order = sel.args.get("order")
    lit = [
        l.this.strip("%").casefold() for l in tree.find_all(exp.Literal) if l.is_string
    ]
    num = [l.this for l in tree.find_all(exp.Literal) if not l.is_string]
    same = sum(
        isinstance(x, exp.EQ) and emit(x.this) == emit(x.expression)
        for x in tree.walk()
    )
    sparse = example["row_count"] == 0 or example["rows"] in ([[0]], [[None]])
    groups = [c.name.lower() for c in grp.find_all(exp.Column)] if grp else []
    group_pk = any(n in cat.pk.get(t, ()) for n in groups for t in schema_names)
    counts = list(tree.find_all(exp.Count))
    distinct = bool(sel.args.get("distinct"))
    return {
        "distance": min(4, projection_distance(tree, example["columns"])),
        "invalid": min(4, invalid),
        "unknown_table": sum(t not in cat.columns for t in schema_names),
        "unused": sum(not refs[t] for t in schema_names),
        "unused_unmentioned": sum(
            not refs[t] and not words(t) & qs for t in schema_names
        ),
        "selfeq": min(same, 3),
        "joins": min(len(joins), 5),
        "unrelated": sum(not words(t) & qs for t in schema_names),
        "mention_t": min(5, len(qs & table_words)),
        "mention_c": min(8, len(qs & cols)),
        "left": sum(bool(j.side) for j in joins),
        "nested": sum(1 for _ in tree.find_all(exp.Subquery)),
        "setop": int(isinstance(tree, exp.SetOperation)),
        "group_pk": int(group_pk),
        "group_len": len(groups),
        "group_order": int(bool(grp and order)),
        "distinct": int(distinct),
        "cue_distinct": int(
            distinct and bool(re.search(r"\b(distinct|different|unique)\b", q))
        ),
        "count_star": sum(isinstance(c.this, exp.Star) for c in counts),
        "count_distinct": sum(isinstance(c.this, exp.Distinct) for c in counts),
        "case": int(bool(tree.find(exp.Lower))),
        "match_literal": sum(bool(x) and x in q for x in lit),
        "extra_literal": sum(x not in q for x in lit),
        "extra_num": sum(x not in q and x not in ("0", "1") for x in num),
        "size": math.log1p(sum(1 for _ in tree.walk())),
        "filters": sum(1 for _ in tree.find_all(exp.Predicate)),
        "shape_joint": prob(lambda p: p == pattern),
        "shape_width": prob(lambda p: p[0] == len(src)),
        "shape_group": prob(lambda p: p[3] != "none") * int(bool(grp)),
        "shape_sort": prob(lambda p: pattern and p[4] == pattern[4]),
        "sparse_extra": int(sparse) * len(joins),
        "sparse_distinct": int(sparse and distinct),
    }


@lru_cache(maxsize=1)
def model():
    return json.loads(Path(__file__).with_name("policy.json").read_text())


def dot(x, weights):
    return sum(v * weights.get(k, 0.0) for k, v in x.items())
