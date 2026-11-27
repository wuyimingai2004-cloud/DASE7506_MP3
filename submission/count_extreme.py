"""Bind a frequency superlative to an entity and a single referencing relation."""
import re
from sqlglot import exp
from repair.edits import parse
from semplan import Schema, child, col, headers, ident, project, roots, words

SUPER = re.compile(r'\b(?:most|(?:greatest|highest|largest|maximum) number(?: of)?)\b', re.I)
EXCLUDE = re.compile(r'''[0-9'"]|\b(?:at (?:least|most)|least|fewest|either|or|except|before|after|without|distinct|different|whose|where|each|per|more than|less than)\b''', re.I)

def proposal(obs):
    q=obs['task']['question']
    m=SUPER.search(q)
    if not m or EXCLUDE.search(q):
        return
    fields=headers(obs['example']['columns'])
    if not fields or not any(not fn for fn,_,_ in fields):
        return
    if any(fn and (fn!='COUNT' or name!='*' or distinct) for fn,name,distinct in fields):
        return
    schema=Schema(obs['history'][2]['observation'])
    root=roots(schema,fields,q[:m.start()])
    if not root:
        return
    tail=q[m.end():]
    # Include inflected event verbs before the superlative ("purchased most often").
    verbs=' '.join(w[:-2] for w in re.findall(r'\b[a-z]{3,}ed\b',q.lower()))
    relation=child(schema,root,tail+' '+verbs,q[:m.start()])
    if not relation:
        return
    table,pk,fk=relation
    sql=(f"SELECT {project(fields,'a')} FROM {ident(root)} AS a JOIN {ident(table)} AS b"
         f" ON {col('a',pk)} = {col('b',fk)} GROUP BY {col('b',fk)} ORDER BY COUNT(*) DESC LIMIT 1")
    return sql,{root.lower(),table.lower()}

def conflict(sql, tables, question):
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or tree.find(exp.Subquery):
        return False
    used={t.name.lower() for t in tree.find_all(exp.Table)}
    order=tree.args.get('order')
    first=order.expressions[0] if order and order.expressions else None
    count=first is not None and first.this.find(exp.Count)
    if first is not None and isinstance(first.this,exp.Column):
        count=count or any(e.alias.lower()==first.this.name.lower() and e.find(exp.Count) for e in tree.expressions)
    extra=used-tables
    return (not tables<=used or not tree.args.get('group') or not count or not first.args.get('desc')
            or bool(extra and all(not words(t)&words(question) for t in extra)))
