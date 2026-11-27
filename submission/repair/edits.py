"""Bounded edits to proposed queries. Public headers are parsed as a narrow grammar."""

import re
from functools import lru_cache

import sqlglot
from sqlglot import exp


@lru_cache(maxsize=1024)
def parse(sql):
    if not isinstance(sql, str) or len(sql.encode()) > 16000:
        return None
    try:
        (tree,) = sqlglot.parse(sql, read="sqlite")
        if not isinstance(tree, (exp.Select, exp.SetOperation)):
            return None
        if any(
            isinstance(node, (exp.DDL, exp.DML, exp.Command, exp.Into))
            for node in tree.walk()
        ):
            return None
        return tree
    except (ValueError, TypeError, RecursionError, sqlglot.errors.SqlglotError):
        return None


def emit(tree):
    return tree.sql(dialect="sqlite", normalize=True, comments=False)


def core(item):
    return item.this if isinstance(item, exp.Alias) else item


def signature(item):
    item = core(item).copy()
    for col in item.find_all(exp.Column):
        col.set("table", None)
    return re.sub(r"\s+", "", emit(item)).casefold()


IDENT = r"[A-Za-z_][\w$]*"
AGG = re.compile(
    rf"(count|avg|sum|max|min|total)\s*\(\s*(distinct\s+)?(?:{IDENT}\.)?({IDENT}|\*)\s*\)",
    re.I,
)


def header(name):
    if re.fullmatch(IDENT, name):
        return exp.column(name)
    m = AGG.fullmatch(name)
    if m:
        # All interpolated tokens matched the restricted grammar above.
        tree = parse(f"SELECT {m[1]}({m[2] or ''}{m[3]})")
        return tree.expressions[0].copy() if tree is not None else None
    return None


def main_select(tree):
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def tables(sel):
    sources = ([sel.args["from"].this] if sel.args.get("from") else []) + [
        j.this for j in sel.args.get("joins", [])
    ]
    return {
        t.alias_or_name.lower(): t.name.lower()
        for t in sources
        if isinstance(t, exp.Table)
    }


def projection(sel, headers, schema):
    """Retain original column qualifications and expand aliases used down-stream."""
    expected = [header(h) for h in headers]
    if any(h is None for h in expected) or not expected:
        return None
    old = sel.expressions
    if [signature(x) for x in old] == [signature(x) for x in expected]:
        return None
    if any(isinstance(x, exp.Star) for x in old):
        return None
    source = tables(sel)
    old_cols = [c for item in old for c in core(item).find_all(exp.Column)]
    available = list(enumerate(old))
    selected = []
    for h in expected:
        idx = next(
            (i for i, item in available if signature(item) == signature(h)), None
        )
        if idx is not None:
            selected.append(old[idx].copy())
            available = [(i, e) for i, e in available if i != idx]
            continue
        for c in h.find_all(exp.Column):
            matches = [p for p in old_cols if p.name.casefold() == c.name.casefold()]
            owners = [
                alias
                for alias, table in source.items()
                if c.name.lower() in schema.get(table, {})
            ]
            if matches:
                if c is h:
                    h = matches[0].copy()
                else:
                    c.replace(matches[0].copy())
            elif owners:
                c.set("table", exp.to_identifier(owners[0]))
            elif schema:
                return None
        selected.append(h)
    out = sel.copy()
    aliases = {item.alias.lower(): core(item) for item in old if item.alias}
    for clause in ("order", "group", "having"):
        if out.args.get(clause):
            for c in list(out.args[clause].find_all(exp.Column)):
                if not c.table and c.name.lower() in aliases:
                    c.replace(aliases[c.name.lower()].copy())
    out.set("expressions", selected)
    return out


def projection_distance(tree, headers):
    parts = main_select(tree).expressions
    if any(isinstance(p, exp.Star) for p in parts):
        return 0
    expected = [header(h) for h in headers]
    return sum(
        i >= len(parts) or (h is not None and signature(parts[i]) != signature(h))
        for i, h in enumerate(expected)
    ) + abs(len(parts) - len(headers))


def rewrite_variants(tree, question, example, schema, config):
    if not isinstance(tree, exp.Select):
        return
    if config["project"]:
        out = projection(tree, example["columns"], schema)
        if out is not None:
            yield out, "projection"
    if not config["local_edits"]:
        return
    base = tree
    if config["project"]:
        base = projection(tree, example["columns"], schema) or tree
    if not base.args.get("group") and not any(
        core(e).find(exp.AggFunc) for e in base.expressions
    ):
        out = base.copy()
        out.set("distinct", None if base.args.get("distinct") else exp.Distinct())
        yield out, "distinct"
    for op in (exp.EQ, exp.NEQ):
        predicates = [
            p
            for p in base.find_all(op)
            if isinstance(p.expression, exp.Literal) and p.expression.is_string
        ]
        if predicates:
            out = base.copy()
            for p in out.find_all(op):
                if isinstance(p.expression, exp.Literal) and p.expression.is_string:
                    p.set("this", exp.Lower(this=p.this.copy()))
                    p.set("expression", exp.Lower(this=p.expression.copy()))
            yield out, "case"
    if base.args.get("group"):
        groups = [
            core(e).copy() for e in base.expressions if not core(e).find(exp.AggFunc)
        ]
        if groups:
            out = base.copy()
            out.set("group", exp.Group(expressions=groups))
            yield out, "group"
