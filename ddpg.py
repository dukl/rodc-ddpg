from globalParameters import GP
from actions import ACT
import os,sys
log_prefix = '[' + os.path.basename(__file__)
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
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][generate action value to be executed]', None, 'optional')
        return 1


