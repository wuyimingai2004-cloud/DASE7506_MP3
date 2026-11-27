"""Single-trajectory SQL repair controller."""
import json
from pathlib import Path
from plain_agent import Agent as Plain
from fallback_agent import Agent as Fallback, compatible, contradicts
from semplan import plans
from repair.controller import load_config
from repair.search import canonical

BASE = Path(__file__).resolve().parent

class Agent:
    def __init__(self, mode):
        self.mode = mode
        self.plain = Plain(load_config(BASE/'plain.json'))
        self.engine = Fallback(load_config(BASE/'fallback.json')) if mode=='ballot' else self.plain
        self.proposals = []
        self.reserved = None
        self.events = {}

    def act(self, obs):
        history = obs['history']
        if len(history) == 3:
            self.prefix = obs
            self.proposals = plans(obs['task']['question'], obs['example']['columns'], history[2]['observation'])
            if self.proposals:
                self.events['has_plan'] = True
                if self.mode in ('route','bridge','handoff','frequency'):
                    self.engine = Fallback(load_config(BASE/'fallback.json'))
                    self.events['routed'] = True
        if self.mode == 'handoff' and not self.proposals and len(history)>=4:
            passing = any(e['observation']['matches_example'] for e in history if e['tool']=='check')
            if len(history)==4 and not passing and history[-1]['tool']=='check':
                self.engine = Fallback(load_config(BASE/'fallback.json'))
                self.engine.pool = self.engine.build(self.prefix)
                self.events['search_handoff'] = True
            elif passing and self.engine is not self.plain:
                # Carry over only the real value-probe state and its locally scored edits.
                self.plain.ground_query = self.engine.ground_query
                for key,value in self.engine.details.items():
                    if key not in self.plain.details:
                        self.plain.details[key] = value
                        self.plain.verified[key] = self.engine.verified.get(key,value)
                self.engine = self.plain
                self.events['returned_to_plain'] = True
        if self.mode=='frequency' and len(history)==3 and not self.proposals:
            from count_extreme import proposal
            self.frequency = proposal(obs)
            if self.frequency:
                self.events['frequency_binding'] = True
        action = self.engine.act(obs)
        if self.mode=='frequency' and getattr(self,'frequency',None):
            from count_extreme import conflict
            sql, tables = self.frequency
            checked={canonical(e['sql']):e for e in history if e['tool']=='check'}
            passing=[e['sql'] for e in checked.values() if e['observation']['matches_example']]
            previous=checked.get(canonical(sql))
            others=[s for s in passing if canonical(s)!=canonical(sql)]
            if (action['tool'] in ('check','submit') and history[-1]['tool']!='execute'
                and all(conflict(s,tables,obs['task']['question']) for s in others)):
                if previous:
                    if action['tool']=='submit' and previous['observation']['matches_example']:
                        self.events['frequency_submit'] = True
                        return dict(tool='submit',sql=previous['sql'])
                elif obs['remaining'] and (action['tool']=='submit' or obs['remaining']==1):
                    if action['tool']=='submit' or conflict(action['sql'],tables,obs['task']['question']):
                        self.events['frequency_check'] = True
                        return dict(tool='check',sql=sql)

        if self.mode == 'ballot':
            if action['tool']=='submit' and not self.proposals:
                passing=[e for e in history if e['tool']=='check' and e['observation']['matches_example']]
                if len(passing)>=2:
                    self.events['ballot'] = True
                    self.plain.pool = self.plain.build(self.prefix)
                    vote = self.plain.tournament(obs,action,passing)
                    if vote!=action:
                        self.events['vote_changed'] = True
                    return vote
            return action
        if self.mode == 'bridge' and not self.proposals and len(history)==3:
            if action['tool']=='check' and not history[0]['observation']['matches_example']:
                tried = canonical(history[0]['sql'])
                eligible = [(self.plain.local[canonical(s)][0],s) for _,s in self.plain.pool.values()
                            if canonical(s)!=tried]
                if eligible:
                    candidate=max(eligible)[1]
                    if canonical(candidate)!=canonical(action['sql']):
                        self.events['first_replaced'] = True
                        return dict(tool='check',sql=candidate)
        if self.mode in ('route','bridge','handoff','frequency') or not self.proposals:
            return action
        checks = [e for e in history if e['tool']=='check']
        passed = [e['sql'] for e in checks if e['observation']['matches_example']]
        if self.reserved:
            if action['tool'] == 'submit' and self.reserved.sql in passed:
                others = [s for s in passed if s != self.reserved.sql]
                if all(contradicts(self.reserved,s) for s in others):
                    self.events['semantic_submit'] = True
                    return dict(tool='submit',sql=self.reserved.sql)
            return action
        if (not obs['remaining'] or action['tool']=='execute' or history[-1]['tool']=='execute'
            or (action['tool']=='check' and obs['remaining']>1)):
            return action
        tried = {canonical(e['sql']) for e in checks}
        proposal = next((p for p in self.proposals if canonical(p.sql) not in tried),None)
        if not proposal or any(not contradicts(proposal,s) for s in passed):
            return action
        if action['tool']=='check' and compatible(proposal,action['sql']):
            return action
        self.events['reserved'] = action['tool']
        self.reserved = proposal
        return dict(tool='check',sql=proposal.sql)

def create_agent(checkpoint=None):
    return Agent(json.loads(Path(checkpoint or BASE/'config.json').read_text())['mode'])
