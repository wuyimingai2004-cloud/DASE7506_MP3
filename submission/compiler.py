"""Bind anonymized reference SQL programs to the current schema, projections and observed literals."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json
import math
import re
from sqlglot import exp
from repair.edits import core, emit, header, main_select, parse, tables
from repair.preferences import stem
from transforms import sources, tags


def records(o):
    return o["history"][2]["observation"]


def affin(typ):
    typ = typ.lower()
    if "int" in typ:
        return "int"
    if re.search("char|text|clob", typ):
        return "text"
    if re.search("real|float|double|numeric|decimal|number", typ):
        return "num"
    if re.search("date|time", typ):
        return "date"
    return "other"


def role(c):
    n = c
    while n.parent is not None and not isinstance(n.parent, exp.Select):
        if isinstance(n.parent, exp.Join):
            return "join"
        n = n.parent
    return n.arg_key or "expressions"


def meta(o):
    schema = records(o)
    return {
        r["name"].lower(): {c["name"].lower(): c for c in r["columns"]} for r in schema
    }


def output_types(names):
    result = []
    for h in map(header, names):
        if h is None:
            return None
        result.append(
            h.key
            + ("/distinct" if h.find(exp.Distinct) else "")
            + ("/star" if h.find(exp.Star) else "")
        )
    return result


def abstract(sql, o):
    tree = parse(sql)
    # Keep a bounded read-only grammar of SELECTs, optionally with set/nested operators.
    if (
        tree is None
        or tree.find(exp.With)
        or tree.find(exp.Anonymous)
        or tree.find(exp.Window)
    ):
        return
    t = tree.copy()
    expected = output_types(o["example"]["columns"])
    if not expected or len(main_select(t).expressions) != len(expected):
        return
    catalog = meta(o)
    owners = []
    columns = []
    literals = []
    descriptors = []
    literal_types = []

    def slot(seq, value):
        if value not in seq:
            seq.append(value)
        return seq.index(value)

    tabnodes = list(t.find_all(exp.Table))
    if not 1 <= len(tabnodes) <= 5 or any(
        n.name.lower() not in catalog for n in tabnodes
    ):
        return
    scopes = {}
    for sel in t.find_all(exp.Select):
        names = tables(sel)
        if (
            not names
            or sel.args.get("from") is None
            or len(names) != 1 + len(sel.args.get("joins", ()))
        ):
            return
        entries = [sel.args["from"].this] + [j.this for j in sel.args.get("joins", ())]
        if any(not isinstance(e, exp.Table) for e in entries):
            return
        scopes[id(sel)] = {
            e.alias_or_name.lower(): (
                e.name.lower(),
                tabnodes.index(e)
                if sum(e == m for m in tabnodes) == 1
                else next(i for i, m in enumerate(tabnodes) if m is e),
            )
            for e in entries
        }
    # First bind actual column roles; names are discarded after abstraction.
    protected = {}
    for j, h in enumerate(main_select(t).expressions):
        cs = list(core(h).find_all(exp.Column))
        if len(cs) == 1:
            protected[id(cs[0])] = j
    for n in t.find_all(exp.Column):
        sel = n.find_ancestor(exp.Select)
        scope = scopes.get(id(sel), {})
        name = n.name.lower()
        if n.table:
            pair = scope.get(n.table.lower())
            if not pair or name not in catalog[pair[0]]:
                return
        else:
            opts = [p for p in scope.values() if name in catalog[p[0]]]
            if len(opts) != 1:
                return
            pair = opts[0]
        table, index = pair
        tab = slot(owners, table)
        col = slot(columns, (tab, name))
        r = catalog[table][name]
        if col == len(descriptors):
            descriptors.append(
                dict(t=tab, aff=affin(r["type"]), pk=bool(r["pk"]), roles=[], heads=[])
            )
        d = descriptors[col]
        if role(n) not in d["roles"]:
            d["roles"].append(role(n))
        if id(n) in protected:
            d["heads"].append(protected[id(n)])
        n.set("this", exp.to_identifier("c" + str(col)))
        n.set("table", exp.to_identifier("a" + str(index)))
    for n in list(t.find_all(exp.Literal)):
        if (
            not n.is_string
            and n.this in ("0", "1")
            and isinstance(n.parent, (exp.Limit, exp.Offset))
        ):
            continue
        key = slot(literals, (n.is_string, n.this))
        if key == len(literal_types):
            literal_types.append("text" if n.is_string else "num")
        n.replace(exp.Literal.string("param" + str(key)))
    for i, n in enumerate(tabnodes):
        index = slot(owners, n.name.lower())
        n.set("alias", exp.TableAlias(this=exp.to_identifier("a" + str(i))))
        n.set("this", exp.to_identifier("r" + str(index)))
    for n in list(t.find_all(exp.Alias)):
        n.replace(n.this.copy())
    # Only anonymous slots are allowed through into a learned program.
    if any(not re.fullmatch(r"[acr]\d+", n.name) for n in t.find_all(exp.Identifier)):
        return
    if t.find(exp.Var) or t.find(exp.DataType) or any(n.comments for n in t.walk()):
        return
    edges = []
    for j in t.find_all(exp.Join):
        on = j.args.get("on")
        if not isinstance(on, exp.EQ) or not all(
            isinstance(c, exp.Column) for c in (on.this, on.expression)
        ):
            return
        edges.append([int(on.this.name[1:]), int(on.expression.name[1:])])
    return dict(
        sql=emit(t),
        cols=descriptors,
        lits=literal_types,
        n=len(owners),
        edges=edges,
        output=expected,
        ordered=o["ordered"],
    )


def fit(rows, obs, gold):
    lib = {}
    for r, o in zip(rows, obs):
        program = abstract(gold[r["id"]], o)
        if program:
            key = json.dumps(program, sort_keys=True, separators=(",", ":"))
            lib.setdefault(key, []).append(sorted(tags(o, records(o))))
    return [[json.loads(k), v] for k, v in lib.items()]


@lru_cache(maxsize=1)
def library():
    doc = json.loads(Path(__file__).with_name("compiler.json").read_text())
    return [(p, [set(v) for v in vs]) for p, vs in doc]


def context(o):
    schema = meta(o)
    hints = Counter()
    roles = defaultdict(Counter)
    values = Counter()
    for s in sources(o):
        t = parse(s)
        if t is None:
            continue
        for n in t.find_all(exp.Table):
            hints[n.name.lower()] += 1
        for c in t.find_all(exp.Column):
            sel = c.find_ancestor(exp.Select)
            if sel is None:
                continue
            opts = tables(sel)
            tab = opts.get(c.table.lower())
            name = c.name.lower()
            for owner in [tab] if tab else opts.values():
                if name in schema.get(owner, {}):
                    roles[role(c)][owner, name] += 1
        for l in t.find_all(exp.Literal):
            if l.parent and not isinstance(l.parent, (exp.Limit, exp.Offset)):
                values["text" if l.is_string else "num", l.this] += 2
    q = o["task"]["question"]
    for val in re.findall(r'["\']([^"\']{1,55})["\']', q):
        values["text", val] += 4
    for val in re.findall(r"\b\d+(?:\.\d+)?\b", q):
        values["num", val] += 3
    for val in list(values):
        if val[1] in q:
            values[val] += 2
    edges = set()
    for r in records(o):
        for e in r["foreign_keys"]:
            if e["from"] and e["to"]:
                a = (r["name"].lower(), e["from"].lower())
                b = (e["table"].lower(), e["to"].lower())
                edges.update(((a, b), (b, a)))
    headers = []
    for z in map(header, o["example"]["columns"]):
        c = z.find(exp.Column) if z is not None else None
        headers.append(c.name.lower() if c is not None else None)
    return schema, hints, roles, values, edges, headers, stem(q)


def instantiate(p, ctx):
    schema, hints, roles, values, edges, headers, q = ctx
    # Beam assigns table slots, pruning with exact public projection column names.
    cols = p["cols"]
    beam = [(0.0, {})]
    for tid in sorted(
        range(p["n"]), key=lambda k: -sum(len(c["heads"]) for c in cols if c["t"] == k)
    ):
        layer = []
        for score, tm in beam:
            for t, names in schema.items():
                if t in tm.values():
                    continue
                heads = [headers[i] for c in cols if c["t"] == tid for i in c["heads"]]
                if any(h not in names for h in heads):
                    continue
                mandatory = [
                    cols[j if cols[i]["t"] == tid else i]["t"]
                    for i, j in p["edges"]
                    if tid in (cols[i]["t"], cols[j]["t"])
                ]
                if any(
                    k in tm and not any(a[0] == t and b[0] == tm[k] for a, b in edges)
                    for k in mandatory
                ):
                    continue
                rank = (
                    score
                    + min(1.5, hints[t] * 0.45)
                    + len(stem(t) & q) / max(1, len(stem(t)))
                )
                layer.append((rank, tm | {tid: t}))
        beam = sorted(layer, key=lambda z: -z[0])[:8]
        if not beam:
            return
    states = [(s, tm, {}) for s, tm in beam]
    for ci in sorted(
        range(len(cols)),
        key=lambda i: (not cols[i]["heads"], "join" not in cols[i]["roles"], i),
    ):
        d = cols[ci]
        layer = []
        for score, tm, cm in states:
            t = tm[d["t"]]
            for n, meta_ in schema[t].items():
                if (t, n) in cm.values():
                    continue
                if any(headers[j] != n for j in d["heads"]):
                    continue
                constrained = True
                for i, j in p["edges"]:
                    if ci not in (i, j):
                        continue
                    other = j if ci == i else i
                    if other in cm and ((t, n), cm[other]) not in edges:
                        constrained = False
                        break
                    other_table = tm[cols[other]["t"]]
                    if not any(a == (t, n) and b[0] == other_table for a, b in edges):
                        constrained = False
                        break
                if not constrained:
                    continue
                w = 0.75 * (affin(meta_["type"]) == d["aff"]) + 0.4 * (
                    bool(meta_["pk"]) == d["pk"]
                )
                w += len(stem(n) & q) / max(1, len(stem(n)))
                w += sum(min(2.0, 0.65 * roles[r][t, n]) for r in d["roles"])
                layer.append((score + w, tm, cm | {ci: (t, n)}))
        states = sorted(layer, key=lambda z: -z[0])[:10]
        if not states:
            return
    lits = [(0.0, [])]
    for typ in p["lits"]:
        choices = [(v, count) for (k, v), count in values.most_common() if k == typ][:4]
        lits = sorted(
            [
                (s + min(2, n * 0.2), vs + [v])
                for s, vs in lits
                for v, n in choices
                if v not in vs
            ],
            key=lambda z: -z[0],
        )[:3]
        if not lits:
            return
    base = parse(p["sql"])
    for score, tm, cm, vs in sorted(
        [(s + a, t, c, b) for s, t, c in states for a, b in lits], key=lambda z: -z[0]
    )[:3]:
        t = base.copy()
        for n in t.find_all(exp.Table):
            n.set("this", exp.to_identifier(tm[int(n.name[1:])], quoted=True))
        for n in t.find_all(exp.Column):
            n.set("this", exp.to_identifier(cm[int(n.name[1:])][1], quoted=True))
        for n in list(t.find_all(exp.Literal)):
            if n.is_string:
                j = int(n.this[5:])
                n.replace(
                    exp.Literal.string(vs[j])
                    if p["lits"][j] == "text"
                    else exp.Literal.number(vs[j])
                )
        yield t


def generate(o):
    output = output_types(o["example"]["columns"])
    qw = tags(o, records(o))
    if not output:
        return
    programs = []
    for p, examples in library():
        if p["output"] != output or p["ordered"] != o["ordered"]:
            continue
        scores = sorted(
            (len(qw & v) / max(1, math.sqrt(len(qw) * len(v))) for v in examples),
            reverse=True,
        )
        similarity = scores[0] + 0.18 * sum(scores[1:4]) / 3
        programs.append((similarity, p))
    ctx = context(o)
    for sim, p in sorted(programs, key=lambda z: -z[0])[:9]:
        if sim >= 0.19:
            for t in instantiate(p, ctx):
                yield t, "compiled", sim
