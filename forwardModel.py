from globalParameters import GP
import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'

class FM:
    def __init__(self):
        pass
    def predict(self, sc, A_avai):
        GP.LOG(log_prefix+'[line-24][predict state s_ts]',None,'optional')