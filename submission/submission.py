"""Route only by the availability of a fully bound scalar-scope edit."""
import json
from pathlib import Path
from local_controller import Agent as Local
from scope_controller import Agent as Scoped
from scope_repair import propose

class Agent:
    def __init__(self,mode):
        self.inner=Local(mode);self.events=self.inner.events
    def act(self,o):
        if len(o['history'])==3 and propose(o):
            replacement=Scoped('scope')
            replacement.base=self.inner.base
            self.inner=replacement;self.events=replacement.events
        return self.inner.act(o)

def create_agent(checkpoint=None):
    return Agent(json.loads(Path(checkpoint or Path(__file__).with_name('config.json')).read_text())['mode'])
