import random
import tensorflow as tf
from globalParameters import GP
from actions import ACT
from observation_reward import OBSRWD
from forwardModel import FM
from ddpg import DDPG
from rodc_ddpg import RODC
from environment import REQ
from dqn_agents import DDQNAgent
import os,sys
import matplotlib.pyplot as plt
import numpy as np
log_prefix = '[' + os.path.basename(__file__)

def feature(lnf, c, lvm):
    diag_lnf = np.diag(np.array(lnf) / GP.maxL)
    diag_cpu = np.diag(np.array(c) / GP.maxC)
    mtx = np.array(lvm) / GP.maxL
    s = GP.k1 * diag_lnf + GP.k2 * diag_cpu + GP.k3 * mtx
    s[s == 0] = 0.000001
    s, _ = np.linalg.eig(s)
    s = np.array([abs(a) for a in s.tolist()])
    s = (s - np.min(s)) / (np.max(s) - np.min(s)) * 255
    return s

class Agent:
    def __init__(self):
        self.ddpg = DDPG()
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-2-3][Initialize actor/critic - behaviour/target networks]', None, 'optional')
        self.forward_model = RODC()
        GP.sess.run(tf.global_variables_initializer())
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-4][Initialize Forward Model]', None, 'optional')
        self.A, self.A_avai = [], []  # RODC-DDPG line-5
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-5][Initialize the action buffer len(A)=%d and len(A_avai)=%d]',(len(self.A),len(self.A_avai)),'optional')
        self.sc = OBSRWD(-1, np.array([0.0001 for _ in range(self.forward_model.state_dim)]), -1)  # latest observation RODC-DDPG Line-6
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-6][Initialize the latest observation sc randomly id(sc)=%d ]', (self.sc.id), 'optional')
        self.T = [[] for _ in range(2)]
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-8][Initialize the raw trajectory T for actions: len(T[0])=%d]', (len(self.T[0])), 'optional')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-8][Initialize the raw trajectory T for observations: len(T[1])=%d]', (len(self.T[1])), 'optional')
        self.D = []
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-8][Initialize the samples for D_fm/D_ddpg: len(D)=%d]', (len(self.D)), 'optional')

        self.requests = []
        self.ue_index = 0
        self.state_dim, self.act_dim = GP.get_dim_action_state()
        self.model = DDQNAgent(self.state_dim, self.act_dim, False, False, 0, 0.001, 0.999, 0.001, 1, False, True)
        self.pending_state, self.pending_action = None, None
        self.step_num = 0


    def reset(self):
        self.requests.clear()
        self.A.clear()
        self.A_avai.clear()
        self.sc = OBSRWD(-1, np.array([0.0001 for _ in range(self.forward_model.state_dim)]), -1)
        self.T = [[] for _ in range(2)]

    def receive_observations(self,obs,ts):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-15][Check: id(sc)=%d, len(A)=%d]',(self.sc.id, len(self.A)),'optional')
        if self.sc.id is -1 and len(self.A) is 0:
            action = ACT(-1, np.array([1.0 for _ in range(self.forward_model.act_dim)]))
            self.A.append(action)
            self.A_avai.append(action)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-16][initialize an action a[%d], len(A)=%d, len(A_avai)=%d]', (action.id,len(self.A),len(self.A_avai)), 'optional')
        else:
            self.T[1].extend(obs)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-18][agent adds observations and rewards into T: len(T[1])=%d]', (len(self.T[1])), 'optional')
            if len(obs) is not 0:
                O_avai = []
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-7][Initialize the received observations O_avai: len(O_avai)=%d]', (len(O_avai)), 'optional')
                for ob in obs:
                    if ob.id > self.sc.id:
                        O_avai.append(ob)
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-19][Check O_avai: len(O_avai)=%d]',(len(O_avai)), 'optional')
                if len(O_avai) is not 0:
                    O_avai.sort(key=lambda OBSRWD: OBSRWD.id, reverse=True)
                    self.sc = O_avai[0]
                    GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-20][agent reset sc = s[%d]]',(self.sc.id), 'optional')
            self.A_avai.clear()
            for act in self.A:
                if act.id >= self.sc.id and act.id < ts:
                    self.A_avai.append(act)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-22][agent reset A_avai: len(A_avai)=%d/ len(A)=%d]',(len(self.A_avai),len(self.A)),'optional')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[agent reset sc to be s[%d] at time step %d]',(self.sc.id, ts), 'data')
        for a in self.A_avai:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[agent reset A_avai to be a[%d] at time step %d]', (a.id, ts), 'data')
        act_value = self.ddpg.act(self.forward_model.predict(self.sc, self.A_avai, ts))
        action = ACT(ts, act_value[0])
        self.A.append(action)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-25][agent adds a[%d] into A: len(A)=%d, len(A_avai)=%d]', (action.id, len(self.A), len(self.A_avai)), 'optional')
        self.T[0].append(action)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-26][agent adds a[%d] into T: len(T[0])=%d]', (action.id, len(self.T[0])), 'optional')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-27][Check if exists adjacent obs]', None, 'optional')
        self.T[1].sort(key=lambda OBSRWD: OBSRWD.id, reverse=False)
        if len(self.T[1]) >= 2:
            remove_eles = [[] for _ in range(2)]
            for index in range(len(self.T[1])-1):
                if self.T[1][index].id + 1 == self.T[1][index+1].id:
                    remove_eles[1].append(self.T[1][index])
                    for act in self.T[0]:
                        if act.id == self.T[1][index].id:
                            remove_eles[0].append(act)
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-28][agent adds samples (s[%d], a[%d], s[%d]) into D_ddpg and D_fm]', (self.T[1][index].id, act.id, self.T[1][index+1].id), 'optional')
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-28][agent adds samples (s[%d], a[%d], s[%d]) into D_ddpg and D_fm]', (self.T[1][index].id, act.id, self.T[1][index+1].id), 'data')
                            #self.D.append([self.T[1][index], act, self.T[1][index+1]])
                            #GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Store one memory: \ns[%d]=%s\na[%d]=%s\nr[%d]=%d\ns[%d]=%s]', (self.T[1][index].id, str(self.T[1][index].value), act.id, str(act.value), self.T[1][index+1].id, self.T[1][index+1].reward, self.T[1][index+1].id, str(self.T[1][index+1].value)), 'procedure')
                            #print(self.T[1][index].value.shape, act.value.shape)
                            self.ddpg.memory.append([self.T[1][index].value.reshape(1, self.forward_model.state_dim), act.value.reshape(1, self.forward_model.act_dim), self.T[1][index+1].reward, self.T[1][index+1].value.reshape(1, self.forward_model.state_dim)])
                            self.forward_model.memory.append([self.T[1][index].value.reshape(1, self.forward_model.state_dim), act.value.reshape(1, self.forward_model.act_dim), self.T[1][index+1].value.reshape(1, self.forward_model.state_dim)])
                            break
            for ele in remove_eles[0]:
                self.T[0].remove(ele)
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-29][agent removes elements a[%d] from T: len(T[0])=%d, len(T[1])=%d]',(ele.id, len(self.T[0]), len(self.T[1])), 'optional')
            for ele in remove_eles[1]:
                self.T[1].remove(ele)
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-29][agent removes elements s[%d] from T: len(T[0])=%d, len(T[1])=%d]', (ele.id, len(self.T[0]), len(self.T[1])), 'optional')
        self.train()
        return action

    def train(self):
        self.ddpg.train()
        self.ddpg.update_target()
        self.forward_model.train()

    def receive_observation_no_delay(self, obs, ts):
        lnf, c, lvm, ret_reward = obs[0].value[0], obs[0].value[1], obs[0].value[2], obs[0].reward
        input_s, actions, img = [], [], []
        req_rate = [0 for _ in range(len(GP.req_type))]
        for req in self.requests:
            req_rate[req.type_id[0]] += 1
        for req in self.requests:
            idx = 1000
            for ms in GP.msc[req.type_id[0]]:
                if ms == 6:
                    continue
                state = feature(lnf, c, lvm)
                img.append(state)
                tmp = np.array([req.type_id[0]+1, req_rate[req.type_id[0]], ms+1])
                input_s.append(np.concatenate((state, tmp), axis=0))
                act = self.model.act(input_s[-1])
                actions.append(act)
                lnf[ms*GP.n_inst+act] += 1
                if idx == 1000:
                    idx = ms*GP.n_inst+act
                else:
                    lvm[idx][ms*GP.n_inst+act] += 1
                    idx = ms*GP.n_inst+act
        self.requests.clear()
        plt.imshow(np.array(img).T, interpolation='nearest', cmap='bone', origin='lower')
        plt.colorbar()
        plt.show()
        #input_s = np.array(input_s)
        #print(input_s)
        #actions = []
        #for s in input_s:
        #    action = self.model.act(s)
        #    print(action)
        #    actions.append(action)
        #if self.pending_state is not None:
        #    self.model.memorize(self.pending_state, self.pending_action, ret_reward, input_s, False)
        #self.pending_state, self.pending_action = input_s, action

        #self.step_num += 1
        #if self.step_num % 200 == 0:
        #    self.model.update_target_model()
        #batch_size = 4
        #if len(self.model.memory) > batch_size and self.step_num % 2 == 0:
        #    batch_loss_dict = self.model.replay(batch_size)

        #GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Generated action: a[%d]=%s]',(ts, str(actions)), 'procedure')
        return ACT(ts, actions)

    def receive_observation_delay_ddpg(self, obs, ts):
        if len(obs) == 0:
            _, act_dim = GP.get_dim_action_state()
            act_value = np.random.rand(act_dim)
            return ACT(ts, act_value)
        else:
            return self.receive_observation_no_delay(obs, ts)

    def check_receive_requests(self, n_newREQs, ts):
        for i in range(n_newREQs):
            index = random.randint(0, GP.n_NF_inst[6] - 1)
            self.requests.append(REQ(i + self.ue_index, 0, 0, 6, index))
        self.ue_index += n_newREQs
