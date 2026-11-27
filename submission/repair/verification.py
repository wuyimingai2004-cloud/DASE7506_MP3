"""Semantic consistency preferences and one bounded database value probe."""

import re
from difflib import SequenceMatcher
from sqlglot import exp
from repair.edits import emit, main_select, parse, tables
from repair.sketch import words


def consistency(sql, obs, x):
    tree = parse(sql)
    sel = main_select(tree)
    src = tables(sel)
    q = obs["task"]["question"].lower()
    qw = words(q)
    score = 0.0
    where = sel.args.get("where")
    if where and isinstance(where.this, exp.And):
        equal = {}
        for cond in where.this.flatten():
            if (
                isinstance(cond, exp.EQ)
                and isinstance(cond.this, exp.Column)
                and isinstance(cond.expression, exp.Literal)
            ):
                key = emit(cond.this)
                val = emit(cond.expression)
                if key in equal and equal[key] != val:
                    score -= 7
                equal[key] = val
    if "both" in qw and isinstance(tree, exp.Intersect):
        score += 2.0
    negative = bool(re.search(r"\b(no|without|never|not|except)\b", q))
    if negative and isinstance(tree, exp.Except):
        score += 1.3
    for node in tree.find_all(exp.Not):
        if isinstance(node.this, exp.In) and node.this.args.get("query"):
            sub = node.this.args["query"].this
            if (
                isinstance(sub, exp.Select)
                and sub.args.get("where")
                and sub.args["where"].find(exp.Is)
                and sub.args["where"].find(exp.Null)
            ):
                score -= 2
    if x["extra_leaf"] and len(src) > 1:
        score -= 1.0 * x["extra_leaf"]
    # Check whether the tables supplying the projection are supported by the
    # noun phrase before the first filtering condition.
    prefix = re.split(
        r"\b(?:where|whose|having|located|with|that have|who have)\b", q, maxsplit=1
    )[0]
    subject = words(prefix)
    for t in {
        src.get(c.table.lower())
        for e in sel.expressions
        for c in e.find_all(exp.Column)
    } - {None}:
        tw = words(t)
        score += 0.75 * len(tw & subject) / max(1, len(tw))
    if re.search(
        r"\b(all (?:the )?(?:data|information|columns|fields)|everything)\b", q
    ):
        score += 2.8 * int(any(e.is_star or e.find(exp.Star) for e in sel.expressions))
    if len({e.sql().lower() for e in sel.expressions}) != len(sel.expressions):
        score -= 1.5
    if sel.args.get("group") and sel.args.get("order") and sel.args.get("limit"):
        term = sel.args["order"].expressions[0].this
        mentions_measure = any(words(c.name) & qw for c in term.find_all(exp.Column))
        if isinstance(term, exp.Sum) and mentions_measure:
            score += 1.0
        if isinstance(term, exp.Count) and re.search(
            r"\b(number of times|most frequent\w*|least frequent\w*|most common|least common)\b",
            q,
        ):
            score += 1.7
    if x.get("gen_legacy"):
        score += 0.45
    if x.get("gen_filter_graph") and not x.get("gen_legacy"):
        score -= 0.75
    if x["order_mismatch"]:
        score -= 1.0
    return score


def value_probe(sqls, obs):
    records = next(e["observation"] for e in obs["history"] if e["tool"] == "schema")
    cols = {
        r["name"].lower(): {c["name"].lower() for c in r["columns"]} for r in records
    }
    q = obs["task"]["question"].lower()
    seen = set()
    for sql in sqls:
        tree = parse(sql)
        if not isinstance(tree, exp.Select) or tree.find(exp.Subquery):
            continue
        for i, p in enumerate(tree.find_all(exp.EQ)):
            if (
                not isinstance(p.this, exp.Column)
                or not isinstance(p.expression, exp.Literal)
                or not p.expression.is_string
            ):
                continue
            term = p.expression.this.strip()
            if not 2 <= len(term) <= 80:
                continue
            name = p.this.name.lower()
            ts = [
                t
                for alias, t in tables(tree).items()
                if name in cols.get(t, ())
                and (not p.this.table or p.this.table.lower() == alias)
            ]
            if len(ts) != 1:
                continue
            ts = ts[0]
            if (ts, name, term) in seen:
                continue
            seen.add((ts, name, term))
            if term.lower() not in q or not re.match(r"[A-Za-z0-9]{2}", term):
                continue
            col = exp.column(name, quoted=True)
            prefix = term.lower()[:2]
            cond = exp.Like(
                this=exp.Lower(this=exp.Trim(this=col.copy())),
                expression=exp.Literal.string(prefix + "%"),
            )
            ask = (
                exp.select(col.copy())
                .distinct()
                .from_(exp.to_table(ts, quoted=True))
                .where(cond)
                .order_by(col.copy())
                .limit(8)
            )
            return emit(ask), tree, i, term


def apply_value(probe, result):
    if not result["ok"]:
        return
    _, tree, index, term = probe
    options = [
        r[0]
        for r in result["rows"]
        if len(r) == 1 and isinstance(r[0], str) and 0 < len(r[0]) < 120
    ]
    if not options or term in options:
        return

    def sim(s):
        a, b = term.lower(), s.strip().lower()
        return max(
            SequenceMatcher(None, a, b).ratio(),
            0.86 if b.startswith(a + " ") else 0.0,
            0.74 if len(b) >= 2 and a.startswith(b) else 0.0,
        )

    best = max(options, key=sim)
    if sim(best) < 0.62:
        return
    out = tree.copy()
    list(out.find_all(exp.EQ))[index].set("expression", exp.Literal.string(best))
    return emit(out)


def diversify(candidate, failures):
    """Penalize a repeated clause structure after a failed real check."""
    from repair.intent import attributes

    if not failures:
        return 0.0
    t = parse(candidate)
    if t is None:
        return 0.0
    a = attributes(t)
    owners = {v.name.lower() for v in t.find_all(exp.Table)}
    close = 0.0
    for other in failures:
        u = parse(other)
        if u is None:
            continue
        b = attributes(u)
        agreement = sum(
            a[k] == b[k]
            for k in ("group", "order", "set", "where", "count", "distinct")
        )
        if owners == {v.name.lower() for v in u.find_all(exp.Table)} and agreement >= 5:
            close = max(close, 0.45 if agreement == 5 else 0.8)
    return close
