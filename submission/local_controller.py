from pathlib import Path
import json
from functools import lru_cache
from sqlglot import exp
from repair.edits import parse
from repair.search import canonical
from frequency_controller import Agent as Frequency
from deltas import proposals,candidates

@lru_cache(maxsize=1)
def rules():
    return json.loads(Path(__file__).with_name('deltas.json').read_text())

def crosses_scope(action,sql):
    src=parse(action['sql']);dst=parse(sql)
    return src is not None and dst is not None and any(src.find(node) and not dst.find(node)
                 for node in (exp.SetOperation,exp.Subquery,exp.Having))

class Agent:
    def __init__(self,mode):
        self.base=Frequency('frequency');self.attempt=None;self.options=None;self.events={}
    def act(self,o):
        a=self.base.act(o);h=o['history']
        if len(h)<3:return a
        checked={canonical(e['sql']):e for e in h if e['tool']=='check'}
        passing={k for k,e in checked.items() if e['observation']['matches_example']}
        if self.attempt:
            if canonical(self.attempt) in passing and a['tool']=='submit':
                self.events['accepted']=True
                return dict(tool='submit',sql=self.attempt)
            return a
        if a['tool']=='execute' or h[-1]['tool']=='execute' or (a['tool']!='submit' and o['remaining']!=1):return a
        if self.options is None:self.options=proposals(o,rules())
        for sql,op,prob,dbs in self.options[:2]:
            if crosses_scope(a,sql):continue
            key=canonical(sql)
            if passing:
                # To challenge a public match, retain the relation, predicates and projection;
                # demand a witnessed parent with the exact coupled grain/measure edit.
                if op not in ('entity_grain+row_measure','output_grain+row_measure') or dbs<2:continue
                parents={canonical(src) for tag,target,src in candidates(o) if tag==op and canonical(target)==key}
                if not parents&passing:continue
                if a['tool']=='submit' and canonical(a['sql']) not in parents|{key}:continue
            if key in checked:
                if key in passing and a['tool']=='submit':
                    self.events.update(accepted=True,edit=op)
                    return dict(tool='submit',sql=checked[key]['sql'])
                continue
            if o['remaining']:
                self.attempt=sql;self.events['edit']=op
                return dict(tool='check',sql=sql)
        return a

def create_agent(checkpoint=None):
    return Agent(json.loads(Path(checkpoint or Path(__file__).with_name('config.json')).read_text())['mode'])
