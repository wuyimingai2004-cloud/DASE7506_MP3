"""Induce short, coupled scope/role edits. The fitted vocabulary contains only operators."""
from collections import Counter, defaultdict
from itertools import product
import re
from sqlglot import exp
from repair.edits import core, emit, parse, header, tables
from repair.relational import Catalog
from repair.search import canonical
from semplan import words

def simple(t):
    return (isinstance(t,exp.Select) and not t.find(exp.Subquery,exp.SetOperation,exp.Window)
            and not t.args.get('with') and isinstance(t.args.get('from'),exp.From)
            and all(isinstance(x,exp.Table) for x in [t.args['from'].this,*[j.this for j in t.args.get('joins',())]])
            and len(tables(t))==len(set(tables(t).values()))==1+len(t.args.get('joins',())))

def owned(c, env, cat):
    if c.table:
        t=env.get(c.table.lower())
        return t if c.name.lower() in cat.columns.get(t,()) else None
    opts=[t for t in env.values() if c.name.lower() in cat.columns.get(t,())]
    return opts[0] if len(opts)==1 else None

def signature(t,cat):
    """Fit-time structural equality: resolve aliases, inner edge ordering, dialect quotes."""
    if not simple(t):return
    env=tables(t)
    def expr(node):
        n=core(node).copy()
        for c in list(n.find_all(exp.Column)):
            t=owned(c,env,cat)
            if t:
                c.set('table',exp.to_identifier(t))
                c.set('this',exp.to_identifier(c.name.lower()))
            elif not c.table and c.this.args.get('quoted'):
                v=exp.Literal.string(c.name)
                if c is n:n=v
                else:c.replace(v)
        return emit(n).lower()
    edges=[]
    for j in t.args.get('joins',()):
        on=j.args.get('on')
        if not isinstance(on,exp.EQ) or j.side:return
        edges.append(tuple(sorted([expr(on.this),expr(on.expression)])))
    return (tuple(sorted(env.values())),tuple(sorted(edges)),tuple(expr(c) for c in t.expressions),
            tuple(expr(t.args[k]) if t.args.get(k) else '' for k in ('distinct','where','group','having','order','limit')))

def scoped_variants(t,o,cat):
    """Copy one SELECT; keep non-edited clauses. Bind header roles, grain and graph together."""
    if not simple(t):return
    env=tables(t); q=o['task']['question'].lower()
    hs=[header(x) for x in o['example']['columns']]
    if not hs or any(h is None for h in hs):return
    if any(j.side or j.kind not in ('','INNER') for j in t.args.get('joins',())):return
    group=t.args.get('group')
    by_name=defaultdict(list)
    for x in t.expressions:
        for c in core(x).find_all(exp.Column):
            u=owned(c,env,cat)
            if u:by_name[c.name.lower()].append(u)
    owners=[]
    for h in hs:
        c=h.find(exp.Column)
        if c is None:owners.append([None]);continue
        name=c.name.lower()
        opts=[u for u in env.values() if name in cat.columns.get(u,())]
        if not opts:return
        owners.append(opts)
    if len(hs)>5 or len(env)>6:return
    count=1
    for opts in owners:count*=len(opts)
    if count>12:return
    reverse={v:k for k,v in env.items()}
    old_sig=[emit(core(c)).lower() for c in t.expressions]
    for binding in product(*owners):
        v=t.copy(); plan=[]
        fixed=[]
        for h,owner in zip(hs,binding):
            n=h.copy(); c=n.find(exp.Column)
            if c is not None:c.set('table',exp.to_identifier(reverse[owner]))
            fixed.append(n)
        moved=any(by_name.get(c.name.lower()) and u not in by_name[c.name.lower()]
                  for h,u in zip(hs,binding) for c in h.find_all(exp.Column))
        # Keep alias references in GROUP/ORDER/HAVING valid after output repair.
        aliases={p.alias.lower():core(p) for p in t.expressions if p.alias}
        for k in ('having','order','group'):
            if v.args.get(k):
                for c in list(v.args[k].find_all(exp.Column)):
                    if not c.table and c.name.lower() in aliases:c.replace(aliases[c.name.lower()].copy())
        v.set('expressions',fixed)
        if moved:plan.append('move_output_owner')
        elif [x.output_name.lower() if isinstance(x,exp.Column) else emit(x).lower() for x in hs]!=[
            core(x).name.lower() if isinstance(core(x),exp.Column) else emit(core(x)).lower() for x in t.expressions]:
            plan.append('bind_output')
        states=[(v,tuple(plan))]
        if group and len(group.expressions)==1:
            dim=[c for c in fixed if isinstance(c,exp.Column)]
            if dim and (v.find(exp.Count) or re.search(r'\b(each|per)\b',q)):
                keys=[('output_grain',dim)]
                if v.args.get('limit') and len({c.table for c in dim})==1:
                    alias=dim[0].table; pks=cat.pk.get(env[alias],())
                    if len(pks)==1:keys.append(('entity_grain',[exp.column(next(iter(pks)),table=alias)]))
                for tag,exprs in keys:
                    g=v.copy();g.set('group',exp.Group(expressions=[c.copy() for c in exprs]))
                    if emit(g.args['group'])!=emit(v.args['group']):states.append((g,tuple(plan+[tag])))
            for r,ops in list(states):
                cnts=[n for k in ('order','having') if r.args.get(k) for n in r.args[k].find_all(exp.Count)
                      if not isinstance(n.this,exp.Star)]
                if cnts and not re.search(r'\b(distinct|different|unique)\b',q):
                    n=r.copy()
                    for k in ('order','having'):
                        if n.args.get(k):
                            for c in n.args[k].find_all(exp.Count):c.set('this',exp.Star())
                    states.append((n,ops+('row_measure',)))
        if v.args.get('order') and not v.args.get('limit') and o['ordered'] and not re.search(
                r'\b(desc\w*|decreas\w*|high\w*|largest|most|recent\w*|reverse|greatest)\b',q):
            if any(e.args.get('desc') for e in v.args['order'].expressions):
                n=v.copy()
                for e in n.args['order'].expressions:
                    e.set('desc',False); e.set('nulls_first',True)
                states.append((n,tuple(plan+['implicit_ascending'])))
        for cur,ops in states:
            if ops:yield ops,cur
            if len(env)<2:continue
            # Rebuild from the actually used owners. Unqualified ambiguity aborts the edit.
            n=cur.copy();n.set('joins',None);n.set('from',None)
            keep=[]
            for c in n.find_all(exp.Column):
                u=owned(c,env,cat)
                if not u:break
                c.set('table',exp.to_identifier(u))
                if u not in keep:keep.append(u)
            else:
                if not keep:continue
                from repair.sketch import from_graph
                repaired,joined=from_graph(n,keep,cat)
                if repaired is None or not joined:continue
                # Equivalent FK graph and owner reassignment alone is already emitted above.
                before=signature(cur,cat); after=signature(repaired,cat)
                if not after or (before and before[:2]==after[:2]):continue
                edge='prune_fk' if set(joined)<set(env.values()) else 'relink_fk'
                yield ops+(edge,),repaired

def candidates(o):
    cat=Catalog(o['history'][2]['observation'])
    sources=list(dict.fromkeys([*o['history'][1]['observation'],o['issue_sql']]))
    for s in sources:
        t=parse(s)
        for ops,v in scoped_variants(t,o,cat):
            # Preserve distinct as given by the parent. Distinctness is a separate repair.
            sql=emit(v)
            if len(sql.encode())<=16000 and canonical(sql)!=canonical(s):
                yield '+'.join(ops),sql,s

def fit(rows,observations,refs):
    counts=defaultdict(Counter)
    domains=defaultdict(set)
    for r,o in zip(rows,observations,strict=True):
        assert r['id']==o['task']['id']
        cat=Catalog(o['history'][2]['observation'])
        target=signature(parse(refs[r['id']]),cat)
        if target is None:continue
        votes=defaultdict(set)
        for op,sql,parent in candidates(o):
            cs=signature(parse(sql),cat)
            if cs==target and signature(parse(parent),cat)!=target:votes[op].add('gain')
            elif cs!=target and signature(parse(parent),cat)==target:votes[op].add('loss')
        for op,labels in votes.items():
            counts[op].update(labels)
            if 'gain' in labels:domains[op].add(r['db_id'])
    return {op:[v['gain'],v['loss'],len(domains[op])] for op,v in sorted(counts.items()) if v['gain']}

def proposals(o,rules):
    out={}
    for op,sql,source in candidates(o):
        if op not in rules:continue
        gain,loss,dbs=rules[op]
        if gain<=loss:continue
        score=(gain+1)/(gain+loss+2)
        # Multiple distinct sources supporting exactly the same edit break ties.
        key=canonical(sql)
        values=out.setdefault(key,[0,0,set(),sql,op])
        if score>values[0]:values[0],values[1],values[4]=score,dbs,op
        values[2].add(canonical(source))
    return [(x[3],x[4],x[0],x[1]) for x in sorted(out.values(),key=lambda x:(-x[0],-x[1],-len(x[2]),x[3]))]
