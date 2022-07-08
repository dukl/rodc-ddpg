from globalParameters import GP
from actions import ACT
import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'
class ACTOR:
    def __init__(self):
        pass

class CRITIC:
    def __init__(self):
        pass

class DDPG:
    def __init__(self):
        self.actor = ACTOR()
        self.critic = CRITIC()
    def act(self, ob):
        GP.LOG(log_prefix+'[line-23][generate action value to be executed]', None, 'optional')
        return 1


