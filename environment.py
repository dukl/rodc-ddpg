from globalParameters import GP
from observation_reward import OBSRWD
import random
import numpy as np


import os, sys
log_prefix = '[' + os.path.basename(__file__)


class ENV:
    def __init__(self):
        self.vms = []
        self.nfs = [[] for _ in range(len(GP.n_NF_inst))]
        self.n_nf_inst = 0
        self.initial_topology()
        self.ues = []
        self.ue_index = 0
        self.it_time = 0
        self.n_msg_req = []
        self.n_msg_reject = []
        self.msg_service_time = []
        self.episode_reward = []
        self.action_reward = -1
        self.old_it_time = 0

    def initial_topology(self):
        for i in range(GP.n_VM):
            self.vms.append(VM(i))
        for i in range(len(GP.n_NF_inst)):
            for j in range(GP.n_NF_inst[i]):
                loc_id = random.randint(0,GP.n_VM-1)
                nf = NF(GP.nf_name[i],loc_id, i, j)
                self.vms[loc_id].nf_instances.append(nf)
                self.nfs[i].append(nf)
                self.n_nf_inst += 1
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Initialize Topology]', None, 'topology')
        for vm in self.vms:
            log_tmp = GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Initialize Topology]' + '[vm-'
            log_tmp += str(vm.id)+'][NF instances: '
            for nf in vm.nf_instances:
                log_tmp += '('+nf.type+','+str(nf.inst_id)+')'
            GP.LOG(log_tmp, None, 'topology')

    def reset(self):
        self.ues.clear()
        self.ue_index = 0
        self.it_time = 0
        self.n_msg_req.clear()
        self.n_msg_reject.clear()
        self.msg_service_time.clear()
        self.action_reward = -1
        self.old_it_time = 0

        for vm in self.vms:
            vm.reset()

        for nf in self.nfs:
            for inst in nf:
                inst.reset()
                log_tmp = GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Reset Signaling Traffic Mapping] inst(' + str(inst.loc_id) + ',' + str(inst.nf_id) + ',' + str(inst.inst_id) + ')--mapping-->\n'
                for i in range(len(inst.next_nf)):
                    if inst.next_nf[i][1] == -1:
                        continue
                    nnf = self.nfs[inst.next_nf[i][1]]
                    sum = 0
                    for ninst in nnf:
                        sum += ninst.C_
                        if len(inst.l_out[i]) < len(nnf):
                            inst.l_out[i].append([ninst.loc_id, ninst.nf_id, ninst.inst_id, 1/len(nnf), 0, 0]) # location, nf, instance, f(l), total, forwarding times
                    for j in range(len(inst.l_out[i])):
                        inst.l_out[i][j][3] = nnf[j].C_/sum
                    max = 0
                    for lo in inst.l_out[i]:
                        if lo[3] > max:
                            max = lo[3]
                    for lo in inst.l_out[i]:
                        lo[4] = int(10/max*lo[3])
                        log_tmp += 'inst(' + str(lo[0]) + ',' + str(lo[1]) + ',' + str(lo[2]) + ',' + str(lo[3]) + ',' + str(lo[4]) + ',' + str(lo[5]) + ')'
                GP.LOG(log_tmp, None, 'data')
    def env_act(self, act_value):
        idx = 0
        for nf in self.nfs:
            for inst in nf:
                log_tmp = GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Reset Signaling Traffic Mapping] inst(' + str(inst.loc_id) + ',' + str(inst.nf_id) + ',' + str(inst.inst_id) + ')--mapping-->\n'
                for i in range(len(inst.next_nf)):
                    if inst.next_nf[i][1] == -1:
                        continue
                    for j in range(len(inst.l_out[i])):
                        inst.l_out[i][j][3] = act_value[idx]
                        idx += 1
                    max = 0
                    for lo in inst.l_out[i]:
                        if lo[3] > max:
                            max = lo[3]
                    if max == 0:
                        max = 0.0001
                    for lo in inst.l_out[i]:
                        lo[4] = int(10/max*lo[3])
                        log_tmp += 'inst(' + str(lo[0]) + ',' + str(lo[1]) + ',' + str(lo[2]) + ',' + str(lo[3]) + ',' + str(lo[4]) + ',' + str(lo[5]) + ')'
                GP.LOG(log_tmp, None, 'data')

    def execute_action(self, action):
        _range = np.max(action.value) - np.min(action.value)
        norm_a = (action.value - np.min(action.value)) / _range
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][env executes a[%d]=%s]',(action.id, str(norm_a)),'procedure')
        self.n_msg_reject.append(0)
        self.msg_service_time.append([0, 0])
        self.env_act(norm_a)
        self.running((action.id + 1)*GP.delta_t)

    def send_obs_reward(self, ts):
        lnf, c, lvm = [], [], [[GP.maxV]*(self.n_nf_inst-1) for _ in range(self.n_nf_inst-1)]
        for nf in self.nfs:
            for inst in nf:
                if inst.nf_id == 6:
                    continue
                if inst.lamda == 0:
                    lnf.append(0.000001)
                else:
                    lnf.append(inst.lamda)
                c.append(inst.C)
                #s.append([inst.loc_id, len(self.vms[inst.loc_id].msg_queue), len(inst.msg_queue), inst.C])
                #GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '(%d,%d,%d)[%d, %d, %d, %d]', (inst.loc_id, inst.nf_id, inst.inst_id, inst.loc_id, len(self.vms[inst.loc_id].msg_queue), len(inst.msg_queue), inst.C), 'optional')
        for i in range(len(self.nfs)):
            if i == 6:
                continue
            for j in range(len(self.nfs[i])):
                for nnf in self.nfs[i][j].next_nf:
                    if nnf[1] == -1:
                        continue
                    for ninst in self.nfs[nnf[1]]:
                        x,y = i*GP.n_inst+j, ninst.nf_id*GP.n_inst+ninst.inst_id
                        if self.nfs[i][j].loc_id == ninst.loc_id:
                            lvm[x][y] = 0
                        else:
                            lvm[x][y] = self.vms[ninst.loc_id].lamda
        s = [lnf, c, lvm]
        obs = OBSRWD(ts, s, self.action_reward)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-12][env sends s[%d]=%s, reward=%f delay=%f]\n', (obs.id, str(s), self.action_reward, obs.n_ts), 'procedure')
        return obs

    def add_dynamics_before_each_time_step(self, next_t):
        self.n_msg_req.append(0)
        for vm in self.vms:
            for msg in vm.msg_queue:
                if msg.ue_id >= 0:
                    self.n_msg_req[-1] += 1
            n_cloud_services = random.randint(0, int(0.2 * GP.mu_VM / 0.01))
            for i in range(n_cloud_services):
                vm.msg_queue.append(REQ(-1, -1, vm.id, -1, -1))
            vm.lamda += n_cloud_services
        for nf in self.nfs:
            for inst in nf:
                self.n_msg_req[-1] += len(inst.msg_queue)
                inst.C_ = GP.nf_cpu_dynamics_sin[int(inst.dynamics_idx + (next_t - GP.delta_t) / 0.01)]
                inst.history_cpu_per_ts.append(inst.C_)
                inst.C = inst.C_
                GP.LOG(
                    GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'time %f-%f, inst-%d-%d-%d has %d CPU cycles',
                    (next_t - GP.delta_t, next_t, inst.loc_id, inst.nf_id, inst.inst_id, inst.C_), 'data')

    def running(self, next_t):
        self.action_reward = -1
        while self.it_time < next_t:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Time Point: %f]', (self.it_time), 'data')
            for vm in self.vms:
                for i in range(GP.mu_VM):
                    if len(vm.msg_queue) > 0:
                        req = vm.msg_queue[0]
                        del vm.msg_queue[0]
                        if req.ue_id < 0:
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Other Cloud Services in VM-%d]  -----> discard',(vm.id), 'data')
                            continue
                        for rise in self.nfs[6]:
                            if vm.id == rise.loc_id:
                                req.start_time = self.it_time
                        msg_to_nf = self.nfs[req.cur_state[1]][req.cur_state[2]]
                        if msg_to_nf.nf_id < 6 and len(msg_to_nf.msg_queue) + 1 > msg_to_nf.l_max:
                            req.is_reject = True
                            self.n_msg_reject[-1] += 1
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[VM-%d sends REQ-%d-%s-%d-UE-%d to NF-%d-%d-%s] - Reject (Overload)', (vm.id, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, msg_to_nf.nf_id, msg_to_nf.inst_id, msg_to_nf.type), 'data')
                        else:
                            msg_to_nf.msg_queue.append(req)
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[VM-%d sends REQ-%d-%s-%d-UE-%d to NF-%d-%d-%s] - Successfully', (vm.id, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, msg_to_nf.nf_id, msg_to_nf.inst_id, msg_to_nf.type), 'request')
            for nf in self.nfs:
                for inst in nf:
                    if inst.nf_processing[0] is not None:
                        if inst.C_ - inst.nf_processing[1] < 0:
                            continue
                        else:
                            inst.C_ -= inst.nf_processing[1]
                            req = inst.nf_processing[0]
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'NF-%d-%d-%d-%s is processed REQ-%d-%s-%d-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')
                            self.send_msg_to_next_nf(inst, req, self.it_time)
                            inst.nf_processing = [None, 0]
                    req = None
                    while (inst.C_ > 0 and len(inst.msg_queue) > 0):
                        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'inst-%d-%d-%d has %d messages in MQ', (inst.loc_id, inst.nf_id, inst.inst_id, len(inst.msg_queue)), 'request')
                        req = inst.msg_queue[0]
                        del inst.msg_queue[0]
                        inst.C_ -= GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]
                        if inst.C_ > 0:
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'NF-%d-%d-%d-%s is processed REQ-%d-%s-%d-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')
                            self.send_msg_to_next_nf(inst, req, self.it_time)
                    if inst.C_ <= 0:
                        inst.nf_processing = [req, -inst.C_]
                        if req is not None:
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'NF-%d-%d-%d-%s is processing REQ-%d-%s-%d-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')
            self.it_time += 0.01
        if self.msg_service_time[-1][1] != 0:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'At time %f~%f, %d REQs are processing, %d REQs are successfully processed (average service time %f), %d REQs are reject; Reward: %f', (next_t-GP.delta_t, next_t, self.n_msg_req[-1], self.msg_service_time[-1][1], self.msg_service_time[-1][0]/self.msg_service_time[-1][1], self.n_msg_reject[-1], (self.msg_service_time[-1][1]/self.n_msg_req[-1])/(self.msg_service_time[-1][0]/self.msg_service_time[-1][1])), 'procedure')
            self.episode_reward[-1] += (self.msg_service_time[-1][1]/self.n_msg_req[-1])/(self.msg_service_time[-1][0]/self.msg_service_time[-1][1])
            self.action_reward = (self.msg_service_time[-1][1]/self.n_msg_req[-1])/(self.msg_service_time[-1][0]/self.msg_service_time[-1][1])
        else:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'At time %f~%f, %d REQs are processing, %d REQs are successfully processed, %d REQs are reject;', (next_t-GP.delta_t, next_t, self.n_msg_req[-1], self.msg_service_time[-1][1], self.n_msg_reject[-1]), 'procedure')
        return 0

    def send_msg_to_next_nf(self, inst, req, it_time):
        if req.cur_loc + 1 >= len(GP.msc[req.type_id[0]]):
            req.end_time = it_time
            #if int(req.end_time) - int(req.start_time) < GP.delta_t:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[REQ-%d-%s-%d-UE-%d has been processed successfully: service time (%f - %f = %f)]', (req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, req.end_time, req.start_time, req.end_time-req.start_time), 'data')
            self.msg_service_time[-1][0] += req.end_time - req.start_time + 0.01
            self.msg_service_time[-1][1] += 1
            nf_rise = self.nfs[6]
            idx = random.randint(0, len(nf_rise) - 1)
            if req.type_id[0] + 1 >= 6:
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[The whole procedure initiated by UE-%d has been processed successfully]',(req.ue_id), 'request')
                self.ues[req.ue_id].end_time = it_time
            else:
                self.n_msg_req[-1] += 1
                self.vms[self.nfs[6][idx].loc_id].msg_queue.insert(0, REQ(req.ue_id, req.type_id[0]+1, nf_rise[idx].loc_id, 6, idx))
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Trigger REQ-%d-%s-%d-UE-%d sending to RISE]', (req.type_id[0]+1, GP.req_type[req.type_id[0]+1], req.cur_loc, req.ue_id), 'request')
            return
        req.cur_loc += 1
        nnf_id = GP.msc[req.type_id[0]][req.cur_loc]
        lout, fwd_idx, index = None, 0, 0
        for i in range(len(GP.next_nf[inst.nf_id])):
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'%d-%d',(GP.next_nf[inst.nf_id][i][1], nnf_id), 'request')
            if GP.next_nf[inst.nf_id][i][1] == nnf_id:
                lout = inst.l_out[i]
                fwd_idx = inst.fwd_idx[i]
                index = i
                break
        if lout[fwd_idx][5] < lout[fwd_idx][4]:
            inst.l_out[index][fwd_idx][5] += 1
        else:
            inst.l_out[index][fwd_idx][5] = 0
            inst.fwd_idx[index] = (inst.fwd_idx[index] + 1) % len(inst.l_out[index])
        ninst = self.nfs[nnf_id][inst.fwd_idx[index]]
        req.cur_state = [ninst.loc_id, ninst.nf_id, ninst.inst_id]
        if ninst.loc_id == inst.loc_id:
            if ninst.nf_id < 6 and len(ninst.msg_queue) + 1 > ninst.l_max:
                #if int(req.end_time) - int(req.start_time) < GP.delta_t:
                self.n_msg_reject[-1] += 1
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[VM-%d sends REQ-%d-%s-%d-UE-%d to NF-%d-%d-%s] - Reject (Overload) - Start-Time: %f', (ninst.loc_id, req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, ninst.nf_id, ninst.inst_id, ninst.type, req.start_time), 'data')

            else:
                ninst.msg_queue.append(req)
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'Send REQ-%d-%s-%d-UE-%d from NF-%d-%d-%d-%s to NF-%d-%d-%d-%s \'s message queue',(req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, inst.loc_id, inst.nf_id, inst.inst_id, inst.type, ninst.loc_id, ninst.nf_id, ninst.inst_id, ninst.type), 'request')
        else:
            self.vms[ninst.loc_id].msg_queue.append(req)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'Send REQ-%d-%s-%d-UE-%d from NF-%d-%d-%d-%s to VM-%d \'s message queue hosting NF-%d-%d-%s',(req.type_id[0], req.type_id[1], req.cur_loc, req.ue_id, inst.loc_id, inst.nf_id, inst.inst_id, inst.type, ninst.loc_id, ninst.nf_id, ninst.inst_id, ninst.type), 'request')

    def update_ue_reqs_every_time_step(self, n_msgs, ts):
        nf_rise = self.nfs[6]
        for i in range(n_msgs):
            index = random.randint(0, len(nf_rise)-1)
            #req_idx = random.randint(0,len(GP.req_type)-1)
            self.vms[self.nfs[6][index].loc_id].msg_queue.append(REQ(i+self.ue_index, 0, nf_rise[index].loc_id, 6, index))
            self.ues.append(UE(i+self.ue_index, ts))
        self.ue_index += n_msgs
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'add %d new %s messages at time %d', (n_msgs, GP.req_type[0], ts), 'data')
        #self.n_msg_req.append(n_msgs)



class VM:
    def __init__(self, id):
        self.L            = 1000  # length of network adapter's queue in a VM
        self.mu           = 0.1   # service time of network adapter in a VM
        self.C            = 1000  # CPU Cycles of a VM
        self.msg_queue    = []    # current queue of network adapter
        self.nf_instances = []    # current 5G-CN NF instances
        self.id           = id
        self.lamda = 0
    def reset(self):
        self.msg_queue.clear()


class NF:
    def __init__(self, type, loc_id, nf_id, inst_id):
        self.type      = type # AMF, SMF, UPF, UDM, UDR, AUSF
        self.loc_id  = loc_id  # located on which VM
        self.l_max     = 100  # current maximum length of queue caused by dynamics
        self.msg_queue = [] # request signaling messages
        #self.C         = random.random()*20000 + 10000  # current CPU cycles
        self.C         = GP.CPU
        self.C_        = self.C
        self.nf_id     = nf_id
        self.inst_id   = inst_id
        self.next_nf   = GP.next_nf[self.nf_id]
        self.l_out     = [[] for _ in range(len(self.next_nf))]  # link out to which instances
        self.fwd_idx   = [0 for _ in range(len(self.next_nf))] # forwarding index
        self.nf_processing = [None,0]
        self.dynamics_idx  = random.randint(100,600)
        self.history_cpu_per_ts = []
        self.lamda = 0
    def reset(self):
        self.msg_queue.clear()
        self.history_cpu_per_ts.clear()
        self.l_out = [[] for _ in range(len(self.next_nf))]
        self.fwd_idx = [0 for _ in range(len(self.next_nf))]
        self.nf_processing = [None, 0]

class REQ:
    def __init__(self, ue_id, type_id, loc_id, nf_id, inst_id):
        self.is_new        = False
        self.is_done       = False
        self.is_reject     = False
        self.is_processing = False
        self.cur_state     = [loc_id, nf_id, inst_id] # loc_id, nf_id, inst_id
        #self.next_nf       = GP.next_nf[self.cur_state[1]]
        self.ue_id         = ue_id
        self.type_id       = [type_id, GP.req_type[type_id]]
        self.cur_loc       = 0
        self.start_time    = 0
        self.end_time      = 0
        if self.ue_id >= 0:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[newly generated REQ: %d, %d-%s]', (self.ue_id, self.type_id[0], self.type_id[1]),'request')
        else:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[newly generated Cloud Services]', None,'request')


class UE:
    def __init__(self, id, it_time):
        self.id         = id
        self.start_time = it_time
        self.end_time   = 0