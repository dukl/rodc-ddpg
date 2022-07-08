from globalParameters import GP
from observation_reward import OBSRWD
import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'

class ENV:
    def __init__(self):
        pass
    def reset(self):
        pass
    def execute_action(self, action):
        GP.LOG(log_prefix+'[line-24][env executes a[%d]]',(action.id),'optional')
    def send_obs_reward(self, ts):
        obs = OBSRWD(ts)
        GP.LOG(log_prefix+'[line-12][env sends s[%d], delay=%f]', (obs.id, obs.n_ts), 'optional')
        return obs