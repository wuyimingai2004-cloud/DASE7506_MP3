import json
from pathlib import Path
from frequency_controller import Agent as Frequency
from repair.search import canonical
from scope_repair import propose,matches

class Agent:
    def __init__(self,mode):
        self.base=Frequency('frequency')
        self.option=None;self.events={}
    def act(self,o):
        h=o['history']
        a=self.base.act(o)
        if len(h)==3:self.option=propose(o)
        if not self.option or a['tool']=='execute' or h[-1]['tool']=='execute':return a
        if a['tool']!='submit' and o['remaining']!=1:return a
        kind,sql,_,_=self.option
        used={canonical(e['sql']):e for e in h if e['tool']=='check'}
        if matches(self.option,a['sql']):return a
        e=used.get(canonical(sql))
        if e:
            if e['observation']['matches_example'] and a['tool']=='submit':
                self.events.update(accepted=True,scope=kind)
                return dict(tool='submit',sql=e['sql'])
        elif o['remaining']:
            self.events['scope']=kind
            return dict(tool='check',sql=sql)
        return a

def create_agent(checkpoint=None):
    return Agent(json.loads(Path(checkpoint or Path(__file__).with_name('config.json')).read_text())['mode'])
