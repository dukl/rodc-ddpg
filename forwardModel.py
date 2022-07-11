from globalParameters import GP
import os,sys
log_prefix = '[' + os.path.basename(__file__)

class FM:
    def __init__(self):
        pass
    def predict(self, sc, A_avai, ts):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][predict state s[%d]]', (ts),'optional')