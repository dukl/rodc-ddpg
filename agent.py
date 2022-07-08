from globalParameters import GP
from actions import ACT
from observation_reward import OBSRWD
from forwardModel import FM
from ddpg import DDPG
import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'

class Agent:
    def __init__(self):
        self.ddpg = DDPG()
        GP.LOG(log_prefix+'[line-2-3][Initialize actor/critic - behaviour/target networks]',None,'optional')
        self.forward_model = FM()
        GP.LOG(log_prefix+'[line-4][Initialize Forward Model]', None, 'optional')
        self.A, self.A_avai = [], []  # RODC-DDPG line-5
        GP.LOG(log_prefix+'[line-5][Initialize the action buffer len(A)=%d and len(A_avai)=%d]',(len(self.A),len(self.A_avai)),'optional')
        self.sc = OBSRWD(-1)  # latest observation RODC-DDPG Line-6
        GP.LOG(log_prefix+'[line-6][Initialize the latest observation sc randomly id(sc)=%d ]', (self.sc.id), 'optional')
        self.T = [[] for _ in range(2)]
        GP.LOG(log_prefix+'[line-8][Initialize the raw trajectory T for actions: len(T[0])=%d]', (len(self.T[0])), 'optional')
        GP.LOG(log_prefix+'[line-8][Initialize the raw trajectory T for observations: len(T[1])=%d]', (len(self.T[1])), 'optional')
        self.D = []
        GP.LOG(log_prefix + '[line-8][Initialize the samples for D_fm/D_ddpg: len(D)=%d]', (len(self.D)), 'optional')


    def reset(self):
        pass
    def receive_observations(self,obs,ts):
        GP.LOG(log_prefix+'[line-15][Check: id(sc)=%d, len(A)=%d]',(self.sc.id, len(self.A)),'optional')
        if self.sc.id is -1 and len(self.A) is 0:
            action = ACT(-1, None)
            self.A.append(action)
            self.A_avai.append(action)
            GP.LOG(log_prefix+'[line-16][initialize an action a[%d], len(A)=%d, len(A_avai)=%d]', (action.id,len(self.A),len(self.A_avai)), 'optional')
        else:
            self.T[1].extend(obs)
            GP.LOG(log_prefix+'[line-18][agent adds observations and rewards into T: len(T[1])=%d]', (len(self.T[1])), 'optional')
            if len(obs) is not 0:
                O_avai = []
                GP.LOG(log_prefix + '[line-7][Initialize the received observations O_avai: len(O_avai)=%d]', (len(O_avai)), 'optional')
                for ob in obs:
                    if ob.id > self.sc.id:
                        O_avai.append(ob)
                GP.LOG(log_prefix+'[line-19][Check O_avai: len(O_avai)=%d]',(len(O_avai)), 'optional')
                if len(O_avai) is not 0:
                    O_avai.sort(key=lambda OBSRWD: OBSRWD.id, reverse=True)
                    self.sc = O_avai[0]
                    GP.LOG(log_prefix+'[line-20][agent reset sc = s[%d]]',(self.sc.id), 'optional')
            self.A_avai.clear()
            for act in self.A:
                if act.id >= self.sc.id and act.id < ts:
                    self.A_avai.append(act)
            GP.LOG(log_prefix+'[line-22][agent reset A_avai: len(A_avai)=%d/ len(A)=%d]',(len(self.A_avai),len(self.A)),'optional')
        act_value = self.ddpg.act(self.forward_model.predict(self.sc, self.A_avai, ts))
        action = ACT(ts, act_value)
        self.A.append(action)
        GP.LOG(log_prefix+'[line-25][agent adds a[%d] into A: len(A)=%d, len(A_avai)=%d]', (action.id, len(self.A), len(self.A_avai)), 'optional')
        self.T[0].append(action)
        GP.LOG(log_prefix+'[line-26][agent adds a[%d] into T: len(T[0])=%d]', (action.id, len(self.T[0])), 'optional')
        GP.LOG(log_prefix+'[line-27][Check if exists adjacent obs]', None, 'optional')
        self.T[1].sort(key=lambda OBSRWD: OBSRWD.id, reverse=False)
        if len(self.T[1]) >= 2:
            remove_eles = [[] for _ in range(2)]
            for index in range(len(self.T[1])-1):
                if self.T[1][index].id + 1 == self.T[1][index+1].id:
                    remove_eles[1].append(self.T[1][index])
                    for act in self.T[0]:
                        if act.id == self.T[1][index].id:
                            remove_eles[0].append(act)
                            GP.LOG(log_prefix+'[line-28][agent adds samples (s[%d], a[%d], s[%d]) into D_ddpg and D_fm]', (self.T[1][index].id, act.id, self.T[1][index+1].id), 'optional')
                            self.D.append([self.T[1][index], act, self.T[1][index+1]])
                            break
            for ele in remove_eles[0]:
                self.T[0].remove(ele)
                GP.LOG(log_prefix + '[line-29][agent removes elements a[%d] from T: len(T[0])=%d, len(T[1])=%d]',(ele.id, len(self.T[0]), len(self.T[1])), 'optional')
            for ele in remove_eles[1]:
                self.T[1].remove(ele)
                GP.LOG(log_prefix + '[line-29][agent removes elements s[%d] from T: len(T[0])=%d, len(T[1])=%d]', (ele.id, len(self.T[0]), len(self.T[1])), 'optional')
        return action