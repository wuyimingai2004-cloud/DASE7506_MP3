"""Move a bound comparison into the appropriate SELECT scope."""
import re
from sqlglot import exp
from repair.edits import parse,emit,header,tables,core
from semplan import words

def sources(o):
    return list(dict.fromkeys([*o['history'][1]['observation'],o['issue_sql']]))

def quantified(o):
    q=o['task']['question'].lower()
    if not re.search(r'\b(every|all|any)\b',q):return
    for sql in sources(o):
        tree=parse(sql)
        if not isinstance(tree,exp.Select):continue
        for cmp in tree.find_all(exp.GT,exp.GTE,exp.LT,exp.LTE):
            quant=cmp.expression
            if not isinstance(quant,(exp.All,exp.Any)) or not isinstance(cmp.this,exp.Column):continue
            sub=quant.this.unnest()
            if not isinstance(sub,exp.Select) or len(sub.expressions)!=1 or sub.args.get('group') or not sub.args.get('where'):continue
            c=core(sub.expressions[0])
            if not isinstance(c,exp.Column) or c.name.lower()!=cmp.this.name.lower():continue
            # Inner filter must actually bind the compared class. An unbound column
            # in a SQLite subquery can otherwise resolve silently to the outer query.
            env=tables(sub)
            schema={r['name'].lower():{x['name'].lower() for x in r['columns']} for r in o['history'][2]['observation']}
            if not all(any((not n.table or n.table.lower()==alias) and n.name.lower() in schema.get(t,())
                       for alias,t in env.items()) for n in sub.find_all(exp.Column)):continue
            if not any(l.is_string and str(l.this).lower() in q for l in sub.args['where'].find_all(exp.Literal)):continue
            # "before ... any" naturally describes a bound before the entire comparison class.
            universal=isinstance(quant,exp.All) or bool(re.search(r'\b(before|after)\b.*\bany\b',q))
            greater=isinstance(cmp,(exp.GT,exp.GTE))
            agg=exp.Max if greater==universal else exp.Min
            n=sub.copy();n.set('expressions',[agg(this=c.copy())])
            out=tree.transform(lambda x:n.subquery() if x is quant or x==quant else x)
            return ('quantifier',emit(out),c.name.lower(),agg.key)

def measure(name,phrase,temporal):
    bits=words(name);qs=words(phrase)
    if temporal:return int(bool(bits&{'date','time','year','timestamp'}))*4
    if 'expensive' in qs or 'cheapest' in qs:
        rank=3*bool(bits&{'cost','price','fare','charge'})
        return rank+int(bool(rank and 'usd' in qs and bits&{'usd','dollar'}))
    return 3*len((qs-{'highest','greatest','largest','smallest','lowest','longest','shortest'})&bits)

def rank_scope(o):
    q=o['task']['question'].lower()
    if not o['ordered'] or o['example']['row_count']>1:return
    m=re.search(r'\b(highest|greatest|largest|smallest|lowest|most expensive|cheapest|most recent(?:ly)?|latest|earliest|longest|shortest)\b',q)
    if not m or re.search(r"['\"0-9]|\b(count|number of|fewest|total|average|sum|each|per|both|either|or|before|after|than|not|without|except|with at least|between)\b",q):return
    hs=[header(s) for s in o['example']['columns']]
    if not hs or not all(isinstance(h,exp.Column) for h in hs):return
    cols=[h.name.lower() for h in hs]
    asc=m[1] in ('smallest','lowest','cheapest','earliest','shortest')
    temp='recent' in m[1] or m[1] in ('latest','earliest')
    hinted=set()
    for sql in sources(o):
        tree=parse(sql)
        if tree is None:continue
        for n in tree.walk():
            child=n.this if isinstance(n,(exp.Max,exp.Min,exp.Ordered)) else None
            if isinstance(child,exp.Column):hinted.add(child.name.lower())
    opts=[]
    for r in o['history'][2]['observation']:
        t=r['name'].lower()
        schema={c['name'].lower():c for c in r['columns']}
        if not set(cols)<=schema.keys():continue
        for name,c in schema.items():
            if c['pk']:continue
            score=measure(name,q[m.start():],temp)
            if not score or (name not in hinted and not temp):continue
            opts.append(((score,int(name in hinted),len(words(t)&words(q))),t,name))
    opts.sort(reverse=True)
    if not opts or len(opts)>1 and opts[0][0]==opts[1][0]:return
    _,t,name=opts[0]
    # Move the scalar evidence into the outer ordering only if all output roles
    # and the metric can be bound in one table. Keep no arbitrary inner filters.
    out=exp.select(*[exp.column(s,quoted=True) for s in cols]).from_(exp.to_table(t,quoted=True))
    out=out.order_by(exp.Ordered(this=exp.column(name,quoted=True),desc=not asc,nulls_first=asc)).limit(1)
    return ('rank',emit(out),name,t)

def propose(o):
    return quantified(o) or rank_scope(o)

def matches(option,sql):
    kind,_,col,role=option;tree=parse(sql)
    if not isinstance(tree,exp.Select):return False
    if kind=='quantifier':
        return any(isinstance(n,(exp.Max,exp.Min)) and n.key==role and isinstance(n.this,exp.Column)
                   and n.this.name.lower()==col for n in tree.find_all(exp.AggFunc))
    order=tree.args.get('order')
    return (set(tables(tree).values())=={role} and order and order.expressions
            and isinstance(order.expressions[0].this,exp.Column) and order.expressions[0].this.name.lower()==col
            and tree.args.get('limit') and not tree.args.get('where') and not tree.args.get('group')
            and bool(order.expressions[0].args.get('desc'))==bool(parse(option[1]).args['order'].expressions[0].args.get('desc')))
