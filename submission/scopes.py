"""Bind predicate arguments inside each side of a set query or each grouping scope."""

import re

from semplan import (
    Plan,
    THRESH,
    OPS,
    NUM,
    choose,
    col,
    ident,
    literal,
    project,
    roots,
    words,
)

COMPARE = re.compile(
    r"\b(at least|at most|more than|fewer than|less than|greater than|higher than|lower than|older than|younger than|above|below|over|under)\s+(-?\d+(?:\.\d+)?)\b",
    re.I,
)
COMPARISONS = OPS | {
    "greater than": ">",
    "higher than": ">",
    "lower than": "<",
    "older than": ">",
    "younger than": "<",
    "above": ">",
    "below": "<",
    "over": ">",
    "under": "<",
}
MEASURES = {
    "older": "age",
    "younger": "age",
    "taller": "height",
    "shorter": "length",
    "longer": "length",
    "heavier": "weight",
}
ROLE_WORDS = {"attendee": "attendance", "aged": "age"}


def terms(s):
    w = words(s)
    return w | {ROLE_WORDS[t] for t in w if t in ROLE_WORDS}


def attribute(schema, t, phrase, kind=None):
    query = terms(phrase) - words(t)
    opts = []
    for c in schema.columns[t].values():
        typ = schema.types[t][c]
        if kind == "text" and any(
            z in typ for z in ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC", "BOOL")
        ):
            continue
        if (
            kind in ("text", "number")
            and (c in schema.pk[t] or re.search(r"(?:_id|ID)$", c))
            and not re.search(r"\bids?\b", phrase, re.I)
        ):
            continue
        tokens = words(c)
        hits = tokens & query
        if hits:
            opts.append((4 * len(hits) + len(hits) / len(tokens), c))
    return choose(opts)


def numeric(schema, t, span):
    """Resolve a measure on either side of its numeric argument within this branch."""
    m = COMPARE.search(span)
    if m:
        op, num = COMPARISONS[m[1].lower()], m[2]
        context = (
            span[: m.start()]
            + " "
            + span[m.end() :]
            + " "
            + MEASURES.get(m[1].lower().split()[0], "")
        )
    else:
        m = re.search(r"(?<![\w.])(-?\d+(?:\.\d+)?)\b", span)
        if not m:
            return
        num, op = m[1], "="
        context = span[: m.start()] + " " + span[m.end() :]
    if len(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?\b", span)) != 1:
        return
    c = attribute(schema, t, context, "number")
    if c:
        return f"{col('a', c)} {op} {num}", c


def set_scopes(schema, cols, q):
    if any(fn for fn, _, _ in cols):
        return
    q = q.rstrip(" ?.!")
    m = re.search(r"\b(both|either|where no|without (?:a|any))\b", q, re.I)
    if not m:
        return
    before, body = q[: m.start()], q[m.end() :].strip()
    t = roots(schema, cols, before)
    if not t:
        return
    select = f"SELECT {project(cols, 'a')} FROM {ident(t)} AS a"
    cue = m[1].lower()
    if cue.startswith(("without", "where")):
        bound = numeric(schema, t, body)
        if bound and words(t) & words(body):
            cond, k = bound
            yield Plan(
                "group_exclusion", f"{select} EXCEPT {select} WHERE {cond}", (t,), (k,)
            )
        return
    intersect = cue == "both"
    segments = re.split(
        r"\band\b" if intersect else r"\bor\b", body, maxsplit=1, flags=re.I
    )
    if len(segments) != 2:
        return
    first, second = segments
    first_pred = numeric(schema, t, first)
    # Each branch gets its own conjunction. A trailing restriction in one arm
    # must not constrain the other arm.
    right = re.split(r"\b(?:but|and)\b", second, flags=re.I)
    predicates = [numeric(schema, t, span) for span in right]
    from_clause = select
    tables = (t,)
    if not (first_pred and all(predicates)):
        if not intersect or any(re.search(r"\d", s) for s in segments):
            return
        pair = text_pair(schema, t, before, first, second)
        if not pair:
            return
        owner, k, v1, v2, edge = pair
        if edge:
            left, right_key = edge
            from_clause += (
                f" JOIN {ident(owner)} AS b ON {col('a', left)} = {col('b', right_key)}"
            )
            tables = (t, owner)
        attr = col("b" if edge else "a", k)
        first_pred = (f"{attr} = {literal(v1)}", k)
        predicates = [(f"{attr} = {literal(v2)}", k)]
    operator = "INTERSECT" if intersect else "UNION"
    sql = (
        f"{from_clause} WHERE {first_pred[0]} {operator} {from_clause} WHERE "
        + " AND ".join(p[0] for p in predicates)
    )
    yield Plan(
        "intersection" if intersect else "disjunction",
        sql,
        tables,
        tuple(sorted({first_pred[1], *(p[1] for p in predicates)})),
    )


def text_pair(schema, root, before, first, second):
    # Values must be literal spans, bound to a named attribute near the coordination.
    # Prefer explicit quotes. For bare values strip a shared attribute suffix only
    # after it has been matched to a real schema column.
    def extract(s):
        quoted = re.search(r"""["']([^"']{1,80})["']""", s)
        if quoted:
            return quoted[1], s[: quoted.start()] + " " + s[quoted.end() :]
        m = re.search(r"\b[A-Z][\w-]*(?:\s+[A-Za-z][\w-]*)*", s)
        if not m:
            return None, ""
        return m[0], s[: m.start()]

    v1, ctx1 = extract(first)
    v2, ctx2 = extract(second)
    if not v1 or not v2:
        return
    last = v2.split()[-1] if len(v2.split()) > 1 else ""
    ctx = " ".join([before.split(" who ")[-1].split(" that ")[-1], ctx1, ctx2, last])
    opts = []
    for t, edge in [(root, None)] + [(t, (a, b)) for t, a, b, _ in schema.edges[root]]:
        k = attribute(schema, t, ctx, "text")
        if not k:
            continue
        wt = words(k)
        score = 4 * len(wt & (terms(ctx) - words(t))) + int(t == root)
        opts.append((score, (t, k, edge or ())))
    picked = choose(opts)
    if not picked:
        return
    t, k, edge = picked

    # The shared column cue may follow one or both literal spans.
    def trim(v):
        parts = v.split()
        if len(parts) > 1 and words(parts[-1]) & words(k):
            return " ".join(parts[:-1])
        return v

    v1, v2 = trim(v1), trim(v2)
    if v1 != v2:
        return t, k, v1, v2, edge


def detail_group(schema, cols, q, threshold):
    if any(f for f, _, _ in cols):
        return
    m = re.search(
        r"\b(?:for each|per)\s+(.+?)\s+(?:who|which|that|with)\b",
        q[: threshold.start()],
        re.I,
    )
    if not m:
        return
    t = roots(schema, cols, q)
    if not t:
        return
    k = attribute(schema, t, m[1])
    if not k or k.lower() in {c.lower() for _, c, _ in cols}:
        return
    num = NUM.get(threshold[2].lower(), threshold[2])
    sql = f"SELECT {project(cols, 'a')} FROM {ident(t)} AS a WHERE {col('a', k)} IN (SELECT {col('b', k)} FROM {ident(t)} AS b GROUP BY {col('b', k)} HAVING COUNT(*) {OPS[threshold[1].lower()]} {num})"
    yield Plan("count_group", sql, (t,), (k,))
