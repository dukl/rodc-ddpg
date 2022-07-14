from globalParameters import GP
from observation_reward import OBSRWD
import random

import os, sys
log_prefix = '[' + os.path.basename(__file__)


class ENV:
    def __init__(self):
        self.vms = []
        self.nfs = [[] for _ in range(len(GP.n_NF_inst))]
        self.n_nf_inst = 0
        self.initial_topology()
        self.request_messages = []
        self.ue_index = 0
        self.it_time = 0
        self.n_msg_req = []
        self.n_msg_reject = []

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
        for nf in self.nfs:
            for inst in nf:
                log_tmp = GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Reset Signaling Traffic Mapping] inst(' + str(inst.loc_id) + ',' + str(inst.nf_id) + ',' + str(inst.inst_id) + ')--mapping-->\n'
                for i in range(len(inst.next_nf)):
                    nnf = self.nfs[inst.next_nf[i][1]]
                    for ninst in nnf:
                        inst.l_out[i].append([ninst.loc_id, ninst.nf_id, ninst.inst_id, round(random.random(),2), 0, 0]) # location, nf, instance, f(l), total, forwarding times
                    max = 0
                    for lo in inst.l_out[i]:
                        if lo[3] > max:
                            max = lo[3]
                    for lo in inst.l_out[i]:
                        lo[4] = int(10/max*lo[3])
                        log_tmp += 'inst(' + str(lo[0]) + ',' + str(lo[1]) + ',' + str(lo[2]) + ',' + str(lo[3]) + ',' + str(lo[4]) + ',' + str(lo[5]) + ')'
                GP.LOG(log_tmp, None, 'topology')

    def execute_action(self, action):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][env executes a[%d]]',(action.id),'optional')
        n_msg_reject = self.running(action.id + GP.delta_t)
        self.n_msg_reject.append(n_msg_reject)

    def send_obs_reward(self, ts):
        obs = OBSRWD(ts)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-12][env sends s[%d], delay=%f]', (obs.id, obs.n_ts), 'optional')
        s = [[] for _ in range(self.n_nf_inst)]
        for nf in self.nfs:
            for inst in nf:
                s.append([inst.loc_id, len(self.vms[inst.loc_id].msg_queue), len(inst.msg_queue), inst.C])
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '(%d,%d,%d)[%d, %d, %d, %d]', (inst.loc_id, inst.nf_id, inst.inst_id, inst.loc_id, len(self.vms[inst.loc_id].msg_queue), len(inst.msg_queue), inst.C), 'observation')
        return obs

    def running(self, next_t):
        n_msg_reject = 0
        while self.it_time < next_t:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Time Point: %f]', (self.it_time), 'data')
            for vm in self.vms:
                if len(vm.msg_queue) > 0:
                    GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[VM-%d has %d messages in MQ - before]',(vm.id, len(vm.msg_queue)), 'data')
                    req = vm.msg_queue[0]
                    del vm.msg_queue[0]
                    GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[VM-%d has %d messages in MQ - after]', (vm.id, len(vm.msg_queue)), 'data')
                    msg_to_nf = self.nfs[req.cur_state[1]][req.cur_state[2]]
                    if msg_to_nf.nf_id < 6 and len(msg_to_nf.msg_queue) + 1 > msg_to_nf.l_max:
                        req.is_reject = True
                        n_msg_reject += 1
                        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[VM-%d sends REQ-%d-%s-UE-%d to NF-%d-%d-%s] - Reject (Overload)', (vm.id, req.type_id[0], req.type_id[1], req.ue_id, msg_to_nf.nf_id, msg_to_nf.inst_id, msg_to_nf.type), 'request')
                    else:
                        msg_to_nf.msg_queue.append(req)
                        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[VM-%d sends REQ-%d-%s-UE-%d to NF-%d-%d-%s] - Successfully', (vm.id, req.type_id[0], req.type_id[1], req.ue_id, msg_to_nf.nf_id, msg_to_nf.inst_id, msg_to_nf.type), 'request')
            for nf in self.nfs:
                for inst in nf:
                    req, cpu_left = inst.nf_processing[0], 0
                    if inst.nf_processing[0] is not None:
                        cpu_left = inst.C - inst.nf_processing[1]
                        if cpu_left > 0:
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'NF-%d-%d-%d-%s is processed REQ-%d-%s-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')
                            self.send_msg_to_next_nf(inst, req)
                    else:
                        cpu_left = inst.C
                    while (cpu_left > 0 and len(inst.msg_queue) > 0):
                        req = inst.msg_queue[0]
                        del inst.msg_queue[0]
                        cpu_left -= GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]
                        if cpu_left > 0:
                            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'NF-%d-%d-%d-%s is processed REQ-%d-%s-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')
                            self.send_msg_to_next_nf(inst, req)
                    if cpu_left <= 0:
                        inst.nf_processing = [req, -cpu_left]
                        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'NF-%d-%d-%d-%s is processing REQ-%d-%s-UE-%d (%d)', (inst.loc_id, inst.nf_id, inst.inst_id, inst.type, req.type_id[0], req.type_id[1], req.ue_id, GP.require_cpu_cycles[req.type_id[0]][inst.nf_id]), 'request')

            self.it_time += 0.01
        return n_msg_reject

    def send_msg_to_next_nf(self, inst, req):
        if req.cur_loc + 1 >= len(GP.msc[req.type_id[0]]):
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[REQ-%d-%s-UE-%d has been processed successfully]', (req.type_id[0], req.type_id[1], req.ue_id), 'data')
            nf_rise = self.nfs[6]
            idx = random.randint(0, len(nf_rise) - 1)
            if req.type_id[0] + 1 >= 6:
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[The whole procedure initiated by UE-%d has been processed successfully]',(req.ue_id), 'data')
            else:
                self.vms[self.nfs[6][idx].loc_id].msg_queue.append(REQ(req.ue_id, req.type_id[0]+1, nf_rise[idx].loc_id, 6, idx))
                GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[Trigger REQ-%d-%s-UE-%d sending to RISE]', (req.type_id[0]+1, GP.req_type[req.type_id[0]+1], req.ue_id), 'data')
            return
        req.cur_loc += 1
        nnf_id = GP.msc[req.type_id[0]][req.cur_loc]
        lout, fwd_idx, index = None, 0, 0
        for i in range(len(GP.next_nf[inst.nf_id])):
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
            ninst.msg_queue.append(req)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'Send REQ-%d-%s-UE-%d from NF-%d-%d-%d-%s to NF-%d-%d-%d-%s \'s message queue',(req.type_id[0], req.type_id[1], req.ue_id, inst.loc_id, inst.nf_id, inst.inst_id, inst.type, ninst.loc_id, ninst.nf_id, ninst.inst_id, ninst.type), 'data')
        else:
            self.vms[ninst.loc_id].msg_queue.append(req)
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + 'Send REQ-%d-%s-UE-%d from NF-%d-%d-%d-%s to VM-%d \'s message queue hosting NF-%d-%d-%s',(req.type_id[0], req.type_id[1], req.ue_id, inst.loc_id, inst.nf_id, inst.inst_id, inst.type, ninst.loc_id, ninst.nf_id, ninst.inst_id, ninst.type), 'data')

    def update_ue_reqs_every_time_step(self, n_msgs):
        nf_rise = self.nfs[6]
        for i in range(n_msgs):
            index = random.randint(0, len(nf_rise)-1)
            self.vms[self.nfs[6][index].loc_id].msg_queue.append(REQ(i+self.ue_index, 0, nf_rise[index].loc_id, 6, index))
        self.ue_index += n_msgs
        self.n_msg_req.append(n_msgs)

class VM:
    def __init__(self, id):
        self.L            = 1000  # length of network adapter's queue in a VM
        self.mu           = 0.1   # service time of network adapter in a VM
        self.C            = 1000  # CPU Cycles of a VM
        self.msg_queue    = []    # current queue of network adapter
        self.nf_instances = []    # current 5G-CN NF instances
        self.id           = id

class NF:
    def __init__(self, type, loc_id, nf_id, inst_id):
        self.type      = type # AMF, SMF, UPF, UDM, UDR, AUSF
        self.loc_id  = loc_id  # located on which VM
        self.l_max     = 100  # current maximum length of queue caused by dynamics
        self.msg_queue = [] # request signaling messages
        self.C         = 10000  # current CPU cycles
        self.nf_id     = nf_id
        self.inst_id   = inst_id
        self.next_nf   = GP.next_nf[self.nf_id]
        self.l_out     = [[] for _ in range(len(self.next_nf))]  # link out to which instances
        self.fwd_idx   = [0 for _ in range(len(self.next_nf))] # forwarding index
        self.nf_processing = [None,0]

class REQ:
    def __init__(self, ue_id, type_id, loc_id, nf_id, inst_id):
        self.is_done       = False
        self.is_reject     = False
        self.is_processing = False
        self.cur_state     = [loc_id, nf_id, inst_id] # loc_id, nf_id, inst_id
        #self.next_nf       = GP.next_nf[self.cur_state[1]]
        self.ue_id         = ue_id
        self.type_id       = [type_id, GP.req_type[type_id]]
        self.cur_loc       = 0
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[newly generated REQ: %d, %d-%s]', (self.ue_id, self.type_id[0], self.type_id[1]),'request')