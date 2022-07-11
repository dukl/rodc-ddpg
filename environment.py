from globalParameters import GP
from observation_reward import OBSRWD
import random

import os, sys
log_prefix = '[' + os.path.basename(__file__)

class ENV:
    def __init__(self):
        self.vms = []
        self.nfs = [[] for _ in range(len(GP.n_NF_inst))]
        self.initial_topology()
    def initial_topology(self):
        for i in range(GP.n_VM):
            self.vms.append(VM(i))
        for i in range(len(GP.n_NF_inst)):
            for j in range(GP.n_NF_inst[i]):
                loc_id = random.randint(0,GP.n_VM-1)
                nf = NF(GP.nf_name[i],loc_id, i, j)
                self.vms[loc_id].nf_instances.append(nf)
                self.nfs[i].append(nf)
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
                        inst.l_out[i].append([ninst.loc_id, ninst.nf_id, ninst.inst_id, 1/len(nnf)])
                        log_tmp += 'inst('+ str(ninst.loc_id) + ',' + str(ninst.nf_id) + ',' + str(ninst.inst_id) + ',' + str(1/len(nnf)) + ')'
                GP.LOG(log_tmp, None, 'topology')




    def execute_action(self, action):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][env executes a[%d]]',(action.id),'optional')
    def send_obs_reward(self, ts):
        obs = OBSRWD(ts)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-12][env sends s[%d], delay=%f]', (obs.id, obs.n_ts), 'optional')
        return obs

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
        self.C         = 0  # current CPU cycles
        self.nf_id     = nf_id
        self.inst_id   = inst_id
        self.next_nf   = GP.next_nf[self.nf_id]
        self.l_out     = [[] for _ in range(len(self.next_nf))]  # link out to which instances