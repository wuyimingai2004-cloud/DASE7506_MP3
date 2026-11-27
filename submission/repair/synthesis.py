"""Compile projections, predicates and set/group cues onto a schema join graph."""

import re
from itertools import islice, product
from sqlglot import exp
from repair.edits import core, emit, header, tables
from repair.sketch import from_graph, words

NEG = re.compile(r"\b(no|not|never|without|neither|except|don't|doesn't|didn't)\b")
EXTREME = re.compile(
    r"\b(most|least|fewest|highest|lowest|largest|smallest|maximum|minimum|longest|shortest|oldest|youngest)\b"
)
COUNT = re.compile(
    r"\b(number|count|each|many|times|common|frequent\w*|fewest|most|least)\b"
)
OPERATORS = (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, exp.Like)
THRESH = re.compile(
    r"\b(more than|fewer than|less than|greater than|at least|at most) (\d+|one|two|three|four|five)\b"
)
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def predicates(trees, question):
    """Recover literal bindings, keeping quoted language and SQL data as literals."""
    out = {}
    for t in trees:
        for p in t.find_all(*OPERATORS):
            if isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal):
                lit = p.expression
                text = str(lit.this).strip("%").lower()
                if not text or text not in question or len(text) > 100:
                    continue
                k = (text, lit.is_string)
                record = out.setdefault(k, [lit.copy(), type(p), set()])
                record[2].add(p.this.name.lower())
    for m in re.finditer(r"""["']([^"']{1,70})["']""", question):
        s = m[1].strip()
        if s and s not in {k[0] for k in out}:
            out[s, True] = [exp.Literal.string(s), exp.EQ, set()]
    return list(out.values())[:3]


def bound_columns(lit, op, hints, question, cat, records):
    text = str(lit.this).strip("%").lower()
    pos = question.find(text)
    before = words(question[max(0, pos - 40) : pos])
    after = words(question[pos + len(text) : pos + len(text) + 22])
    options = []
    for r in records:
        t = r["name"].lower()
        for c in r["columns"]:
            n = c["name"].lower()
            if not n or n == "*":
                continue
            w = words(n)
            typ = c["type"].upper()
            if (
                lit.is_string
                and any(a in typ for a in ("INT", "REAL", "NUMBER", "FLOAT"))
                and n not in hints
            ):
                continue
            score = (
                2.5 * (n in hints)
                + 2 * len(w & before)
                + 0.7 * len(w & after)
                + 0.4 * len(words(t) & words(question))
                - 0.45 * max(0, len(w) - len(w & before) - len(w & after))
                - 0.8 * bool(c["pk"])
            )
            if score > 0:
                options.append(
                    (
                        score,
                        t,
                        op(
                            this=exp.column(n, table=t, quoted=True),
                            expression=lit.copy(),
                        ),
                    )
                )
    return sorted(options, key=lambda a: (-a[0], a[1], emit(a[2])))[:3]


def generate(trees, obs, cat, raw):
    example = obs["example"]
    q = obs["task"]["question"].lower()
    headers = [header(h) for h in example["columns"]]
    if not headers or any(h is None for h in headers):
        return
    qs = words(q)
    negative = bool(NEG.search(q))
    extreme = EXTREME.search(q)
    threshold = THRESH.search(q)
    sorted_result = obs["ordered"]
    hinted = {t.name.lower() for tree in trees for t in tree.find_all(exp.Table)}
    ranking = sorted(
        cat.columns, key=lambda t: (-len(words(t) & qs), t not in hinted, t)
    )
    mentioned = [t for t in ranking if words(t) & qs]
    if not ranking:
        return
    params = predicates(trees, q)
    options = [bound_columns(*p, q, cat, raw) for p in params]
    options = [x for x in options if x]
    bindings = []
    for h in headers:
        cols = list(h.find_all(exp.Column))
        if len(cols) > 1:
            return
        if not cols:
            bindings.append([(h, None)])
            continue
        c = cols[0].name.lower()
        owners = [t for t in ranking if c in cat.columns[t]]
        if not owners:
            return
        layer = []
        for t in owners[:3]:
            p = h.copy()
            a = next(p.find_all(exp.Column))
            a.set("table", exp.to_identifier(t, quoted=True))
            layer.append((p, t))
        bindings.append(layer)

    def compile_sql(project, tabs, conds=()):
        stmt, joined = from_graph(
            exp.select(*(p.copy() for p in project)), list(dict.fromkeys(tabs)), cat
        )
        if stmt is not None and conds:
            stmt = stmt.where(exp.and_(*(v.copy() for v in conds)))
        return stmt

    # Few alternatives per clause, with high lexical evidence first.
    combos = sorted(
        islice(product(*bindings), 512), key=lambda c: len(set(t for _, t in c if t))
    )[:3]
    for combo in combos:
        project = [p for p, _ in combo]
        roots = list(dict.fromkeys(t for _, t in combo if t))
        start = roots or (mentioned or ranking)[:1]
        base = compile_sql(project, start)
        if base is None:
            continue
        if not sorted_result and not options and len(start) == 1:
            yield base, "plain", 0.4
            if not roots:
                for t in (mentioned or ranking)[1:3]:
                    yield compile_sql(project, [t]), "plain", 0.15
        if (
            not sorted_result
            and len(project) == 1
            and roots
            and isinstance(project[0], exp.Column)
            and re.search(r"\b(and|or)\b", q)
        ):
            name = project[0].name.lower()
            other = [
                t
                for t in mentioned
                if t != roots[0]
                and name in cat.columns[t]
                and (words(t) - words(roots[0])) & qs
            ]
            if roots[0] in mentioned and other:
                b = exp.select(exp.column(name, table=other[0], quoted=True)).from_(
                    exp.to_table(other[0], quoted=True)
                )
                yield base.union(b), "entity_union", 1.4
        if options:
            for vals in list(product(*options))[:9]:
                pts = [v[1] for v in vals]
                preds = [v[2] for v in vals]
                sel = compile_sql(project, start + pts, preds)
                if sel is None:
                    continue
                if negative:
                    yield base.except_(sel), "negative", 1.1
                elif len(vals) == 2 and re.search(r"\b(both|either|or)\b", q):
                    a = compile_sql(project, start + [pts[0]], [preds[0]])
                    b = compile_sql(project, start + [pts[1]], [preds[1]])
                    if a is not None and b is not None:
                        yield (
                            a.intersect(b) if "both" in q else a.union(b),
                            "split_filter",
                            1.0,
                        )
                elif not sorted_result:
                    yield sel, "filter_graph", 0.3
        if not roots:
            continue
        extra = [
            t
            for t in mentioned
            if t not in start
            and (words(t) - set().union(*(words(r) for r in start))) & qs
        ]
        anchors = [start] + [start + [t] for t in extra[:2]]
        display = [p for p in project if isinstance(p, exp.Column)]
        if not display:
            continue
        has_count = any(p.find(exp.Count) for p in project)
        for tabs in anchors:
            sel = compile_sql(project, tabs)
            if sel is None:
                continue
            if negative and len(tabs) > len(start) and not options:
                yield base.except_(sel), "negative", 0.6
            for group in [
                display,
                [
                    exp.column(k, table=roots[0], quoted=True)
                    for k in sorted(cat.pk.get(roots[0], ()))
                ],
            ]:
                if not group:
                    continue
                g = sel.group_by(*(p.copy() for p in group))
                if has_count and not sorted_result:
                    yield g, "entity_count", 1.0
                if threshold and not sorted_result:
                    op = {
                        "more than": exp.GT,
                        "greater than": exp.GT,
                        "at least": exp.GTE,
                        "at most": exp.LTE,
                        "less than": exp.LT,
                        "fewer than": exp.LT,
                    }[threshold[1]]
                    num = exp.Literal.number(NUM.get(threshold[2], threshold[2]))
                    yield (
                        g.having(op(this=exp.Count(this=exp.Star()), expression=num)),
                        "count_threshold",
                        1.1,
                    )
                if extreme and sorted_result and COUNT.search(q):
                    desc = extreme[1] not in (
                        "least",
                        "fewest",
                        "lowest",
                        "smallest",
                        "minimum",
                        "shortest",
                        "youngest",
                    )
                    yield (
                        g.order_by(
                            exp.Ordered(this=exp.Count(this=exp.Star()), desc=desc)
                        ).limit(1),
                        "count_extreme",
                        1.0,
                    )
