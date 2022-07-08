from globalParameters import GP
from actions import ACT
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
        GP.LOG('[DDPG][RODC-DDPG][line-23][generate action value to be executed]', None, 'optional')
        return 1


