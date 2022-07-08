from globalParameters import GP
from actions import ACT
from observation_reward import OBSRWD
from forwardModel import FM
from ddpg import DDPG
class Agent:
    def __init__(self):
        self.ddpg = DDPG()
        GP.LOG('[Agent][RODC-DDPG][line-2-3][Initialize actor/critic - behaviour/target networks]',None,'optional')
        self.forward_model = FM()
        GP.LOG('[Agent][RODC-DDPG][line-4][Initialize Forward Model]', None, 'optional')
        self.A, self.A_avai = [], []  # RODC-DDPG line-5
        GP.LOG('[Agent][RODC-DDPG][line-5][Initialize the action buffer len(A)=%d and len(A_avai)=%d]',(len(self.A),len(self.A_avai)),'optional')
        self.sc = OBSRWD(-1)  # latest observation RODC-DDPG Line-6
        GP.LOG('[Agent][RODC-DDPG][line-6][Initialize the latest observation sc randomly id(sc)=%d ]', (self.sc.id), 'optional')


    def reset(self):
        pass
    def receive_observations(self,obs,ts):
        GP.LOG('[Agent][RODC-DDPG][line-14][Check: id(sc)=%d, len(A)=%d]',(self.sc.id, len(self.A)),'optional')
        if self.sc.id is -1 and len(self.A) is 0:
            action = ACT(-1, None)
            self.A.append(action)
            self.A_avai.append(action)
            GP.LOG('[Agent][RODC-DDPG][line-15][initialize an action a[%d], len(A)=%d, len(A_avai)=%d]', (action.id,len(self.A),len(self.A_avai)), 'optional')
        act_value = self.ddpg.act(self.forward_model.predict(self.sc, self.A_avai))
        return ACT(ts, act_value)