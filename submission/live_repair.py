"""One bounded diagnostic on a literal-bearing column, or a repair supported by actual check output."""

from difflib import SequenceMatcher
import re
from sqlglot import exp
from repair.edits import core, emit, main_select, parse, tables
from repair.search import canonical
from repair.preferences import stem


def quote_ident(name):
    return exp.to_identifier(name, quoted=True).sql(dialect="sqlite")


def similarity(left, right):
    def normalize(s):
        return "".join(re.findall(r"[a-z0-9]+", s.casefold()))

    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    abbr = "".join(w[0] for w in re.findall(r"[a-z0-9]+", left.casefold()))
    if abbr == b and len(b) > 1:
        return 0.93
    if min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)):
        return 0.74 + 0.14 * min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def propose_probe(sql, o):
    t = parse(sql)
    if t is None:
        return
    cat = {
        r["name"].lower(): {c["name"].lower() for c in r["columns"]}
        for r in o["history"][2]["observation"]
    }
    options = []
    for j, l in enumerate(t.find_all(exp.Literal)):
        if not l.is_string or not 2 <= len(l.this) <= 64:
            continue
        pred = l.parent
        if not isinstance(pred, (exp.EQ, exp.NEQ, exp.In, exp.Like)):
            continue
        col = pred.this
        if not isinstance(col, exp.Column):
            continue
        sel = col.find_ancestor(exp.Select)
        if sel is None:
            continue
        env = tables(sel)
        own = [
            owner
            for alias, owner in env.items()
            if (not col.table or col.table.lower() == alias)
            and col.name.lower() in cat.get(owner, ())
        ]
        if len(own) != 1:
            continue
        term = l.this.strip("% ").strip()
        if len(term) < 2 or len(term) > 60:
            continue
        # Probe only one real column. Prefer literals explicitly anchored to the question.
        confidence = (
            1.0 if term.casefold() in o["task"]["question"].casefold() else 0.45
        )
        options.append((confidence, j, term, col.name, own[0]))
    if not options:
        return
    _, j, term, col, tab = max(options)
    ident = quote_ident(col)
    prefix = exp.Literal.string(term[:2].lower() + "%").sql(dialect="sqlite")
    first = exp.Literal.string(term[:1].lower() + "%").sql(dialect="sqlite")
    # Natural frequency breaks ties among prefix matches. A single real execute returns at most eight values.
    diagnostic = (
        f"SELECT {ident} FROM {quote_ident(tab)} WHERE {ident} IS NOT NULL GROUP BY {ident} "
        f"ORDER BY CASE WHEN LOWER(TRIM({ident})) LIKE {prefix} THEN 0 "
        f"WHEN LOWER(TRIM({ident})) LIKE {first} THEN 1 ELSE 2 END, COUNT(*) DESC, {ident} LIMIT 8"
    )
    return diagnostic, sql, j, term


def apply_probe(p, fb):
    if not fb.get("ok"):
        return
    _, sql, j, term = p
    options = [
        r[0]
        for r in fb["rows"]
        if len(r) == 1 and isinstance(r[0], str) and 0 < len(r[0]) < 150
    ]
    if not options or term in options:
        return
    score, value = max((similarity(term, s), s) for s in options)
    if score < 0.64:
        return
    out = parse(sql).copy()
    node = list(out.find_all(exp.Literal))[j]
    if isinstance(node.parent, exp.Like):
        value = (
            ("%" if node.this.startswith("%") else "")
            + value
            + ("%" if node.this.endswith("%") else "")
        )
    node.replace(exp.Literal.string(value))
    return emit(out)


def multiplicity(sql, fb, o):
    if not fb.get("ok"):
        return
    t = parse(sql)
    if not isinstance(t, exp.Select) or t.find(exp.Subquery):
        return
    rows = fb["rows"]
    ex = o["example"]
    if len(ex["columns"]) != len(fb["columns"]):
        return
    if (
        not fb["truncated"]
        and not ex["truncated"]
        and len(rows) != ex["row_count"]
        and set(map(tuple, rows)) == set(map(tuple, ex["rows"]))
    ):
        v = t.copy()
        if len(rows) > ex["row_count"] and not v.args.get("distinct"):
            v.set("distinct", exp.Distinct())
            yield emit(v)
        elif len(rows) < ex["row_count"] and v.args.get("distinct"):
            v.set("distinct", None)
            yield emit(v)
    if (
        t.args.get("group")
        or len(t.expressions) != 1
        or len(rows) != 1
        or len(ex["rows"]) != 1
    ):
        return
    count = core(t.expressions[0])
    if (
        not isinstance(count, exp.Count)
        or not isinstance(rows[0][0], int)
        or not isinstance(ex["rows"][0][0], int)
    ):
        return
    if rows[0][0] <= ex["rows"][0][0] or not t.args.get("joins"):
        return
    env = tables(t)
    schema = {r["name"].lower(): r for r in o["history"][2]["observation"]}
    q = stem(o["task"]["question"])
    for alias, table in sorted(env.items(), key=lambda z: -len(stem(z[1]) & q)):
        pk = [c["name"] for c in schema.get(table, {}).get("columns", ()) if c["pk"]]
        if len(pk) == 1:
            v = t.copy()
            core(v.expressions[0]).set(
                "this",
                exp.Distinct(expressions=[exp.column(pk[0], table=alias, quoted=True)]),
            )
            yield emit(v)


def action(agent, o, checks):
    h = o["history"]
    used = {canonical(e["sql"]) for e in checks}
    if (
        h[-1]["tool"] == "execute"
        and getattr(agent, "diagnostic", None)
        and h[-1]["sql"] == agent.diagnostic[0]
    ):
        new = apply_probe(agent.diagnostic, h[-1]["observation"])
        if new and canonical(new) not in used and len(new.encode()) <= 16000:
            return dict(tool="check", sql=new)
    if not o["remaining"] or any(e["observation"]["matches_example"] for e in checks):
        return
    if o["remaining"] > 3:
        return
    for e in reversed(checks):
        for new in multiplicity(e["sql"], e["observation"], o):
            if canonical(new) not in used and len(new.encode()) <= 16000:
                return dict(tool="check", sql=new)
    if o["remaining"] >= 2 and not getattr(agent, "diagnostic", None):
        for e in reversed(checks):
            if not e["observation"]["ok"]:
                continue
            p = propose_probe(e["sql"], o)
            if p and not any(
                a.get("sql") == p[0] and a["tool"] == "execute" for a in h
            ):
                agent.diagnostic = p
                return dict(tool="execute", sql=p[0])
