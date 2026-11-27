"""Optional one-shot, read-only value grounding via the official execute action."""

import re
from difflib import SequenceMatcher
from sqlglot import exp
from repair.edits import emit, parse, tables


def anchor_probe(attempts, obs):
    if obs["example"]["row_count"] == 0:
        return None
    records = next(
        (e["observation"] for e in obs["history"] if e["tool"] == "schema"), []
    )
    schema = {
        t["name"].lower(): {c["name"].lower() for c in t["columns"]} for t in records
    }
    for _, (sql, feedback) in reversed(list(attempts.items())):
        if not feedback["ok"] or (
            feedback["row_count"] and feedback["rows"] not in ([[0]], [[None]])
        ):
            continue
        tree = parse(sql)
        if not isinstance(tree, exp.Select) or tree.find(exp.Subquery):
            continue
        owners = tables(tree)
        for idx, p in enumerate(tree.find_all(exp.EQ)):
            if (
                not isinstance(p.this, exp.Column)
                or not isinstance(p.expression, exp.Literal)
                or not p.expression.is_string
            ):
                continue
            term = p.expression.this
            if (
                len(term) > 80
                or not term.strip()
                or term.lower() not in obs["task"]["question"].lower()
            ):
                continue
            name = p.this.name.lower()
            ts = [
                owner
                for alias, owner in owners.items()
                if name in schema.get(owner, set())
                and (not p.this.table or alias == p.this.table.lower())
            ]
            if len(ts) != 1:
                continue
            prefix = next(iter(re.findall(r"[A-Za-z0-9]+", term)), None)
            if not prefix:
                continue
            col = exp.column(name, quoted=True)
            sample = (
                exp.select(col.copy())
                .distinct()
                .from_(exp.to_table(ts[0], quoted=True))
            )
            match = exp.Like(
                this=exp.Lower(this=col.copy()),
                expression=exp.Literal.string(prefix[:8].lower() + "%"),
            )
            probe = emit(sample.where(match).order_by(col.copy()).limit(6))
            return probe, tree, idx, term
    return None


def repair_from_probe(tree, index, term, result):
    if not result["ok"]:
        return None
    options = [
        r[0]
        for r in result["rows"]
        if len(r) == 1
        and isinstance(r[0], str)
        and 0 < len(r[0]) < 128
        and r[0] != term
    ]

    def sim(s):
        left, right = term.lower().strip(), s.lower().strip()
        return max(
            SequenceMatcher(None, left, right).ratio(),
            0.83 if right.startswith(left + " ") else 0,
        )

    if options:
        best = max(options, key=sim)
        if sim(best) >= 0.65:
            out = tree.copy()
            list(out.find_all(exp.EQ))[index].set(
                "expression", exp.Literal.string(best)
            )
            return emit(out)
    return None
