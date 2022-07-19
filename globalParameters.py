import numpy as np
from logger import log_debug,log_data, log_procd
import os, sys, math
log_prefix = '[' + os.path.basename(__file__)
class GP:
    # System Parameters
    n_episode = 2
    n_time_steps = 100
    delta_t   = 1
    @staticmethod
    def RESET(env, agent):
        env.reset()
        agent.reset()
        GP.generate_dynamics_sin()

    # System Model
    CPU = 200000
    n_UEs = 1000
    n_VM = 8 # number of VMs
    n_NF_inst = [3,2,2,2,2,2,1] # number of instances of AMF, SMF, UPF, UDM, UDR, AUSF, RISE
    nf_name   = ["AMF", "SMF", "UPF", "UDM", "UDR", "AUSF", "RISE"]
    next_nf   = [[["SMF",1],["AUSF",5]], [["AMF",0], ["UPF",2]], [["SMF",1]], [["UDR",4]], [["",-1]], [["UDM",3]],[["AMF", 0]]]
    req_type  = ["RegistrationRequest","AuthenticationResponse","SecurityModeComplete","IdentityResponse","RegistrationComplete","PDUSessionEstablishmentRequest"]
    require_cpu_cycles = [[1200,0,0,2000,500,1500,10], [1000,0,0,800,10,850,10],[1000,0,0,0,0,0,10],[1000,0,0,0,0,0,10],[1000,0,0,0,0,0,10],[1200,2000,2500,0,0,0,10]]
    msc       = [[6,0,5,3,4], [6,0,5,3,4], [6,0], [6,0], [6,0], [6,0,1,2]]
    @staticmethod
    def get_dim_action_state():
        state_dim, act_dim = 0, 0
        for num in GP.n_NF_inst:
            state_dim += num
        state_dim *= 4
        for i in range(len(GP.nf_name)):
            for nnf in GP.next_nf[i]:
                if nnf[1] == -1:
                    continue
                act_dim += GP.n_NF_inst[i]*GP.n_NF_inst[nnf[1]]

        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[state_dim = %d, action_dim = %d]', (state_dim, act_dim), 'optional')
        return state_dim, act_dim

    # rate in VM
    mu_VM = 1

    # Log setting
    logDebugAvai = False
    logDataAvai  = False
    logOptional  = False
    logTopology  = False
    logRequest   = False
    logObservation   = False
    logProcedure = True
    @staticmethod
    def getLogInfo(log_prefix, line):
        return log_prefix +'-' +str(line) +'][RODC-DDPG]'
    def LOG(str, value, type):
        if GP.logDebugAvai and type is 'debug':
            if value is not None:
                log_debug.logger.debug(str % value)
            else:
                log_debug.logger.debug(str)
        if GP.logDataAvai and type is 'data':
            if value is not None:
                log_data.logger.debug(str % value)
            else:
                log_data.logger.debug(str)
        if GP.logOptional and type is 'optional':
            if value is not None:
                log_debug.logger.debug(str % value)
            else:
                log_debug.logger.debug(str)
        if GP.logTopology and type is 'topology':
            if value is not None:
                log_debug.logger.debug(str % value)
            else:
                log_debug.logger.debug(str)
        if GP.logRequest and type is 'request':
            if value is not None:
                log_debug.logger.debug(str % value)
            else:
                log_debug.logger.debug(str)
        if GP.logRequest and type is 'observation':
            if value is not None:
                log_data.logger.debug(str % value)
            else:
                log_data.logger.debug(str)
        if GP.logProcedure and type is 'procedure':
            if value is not None:
                log_procd.logger.debug(str % value)
            else:
                log_procd.logger.debug(str)

    # System Running
    obs_on_road = []
    @staticmethod
    def CHECK_OBSERVATIONS(ts):
        O= []
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-7][Initialize the received observations O: len(O)=%d]',(len(O)), 'optional')
        for ob in GP.obs_on_road:
            if ob.id + ob.n_ts <= ts:
                O.append(ob)
                GP.obs_on_road.remove(ob)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-13][received observations len(O)=%d]',len(O),'optional')
        for ob in O:
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[agent receives observation s[%d] at time %d]', (ob.id, ts), 'data')
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-13][received observation s[%d]]', ob.id, 'optional')
        return O

    # Dynamics simulation
    nf_cpu_change_rate_sin = 1
    nf_cpu_dynamics_sin = None
    @staticmethod
    def generate_dynamics_sin():
        x = np.arange(0, 3*GP.n_time_steps*GP.delta_t, 0.01)
        GP.nf_cpu_dynamics_sin = np.sin(GP.nf_cpu_change_rate_sin * x) * GP.CPU * GP.delta_t + GP.CPU * GP.delta_t

