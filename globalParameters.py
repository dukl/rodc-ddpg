from logger import log_debug,log_data
import os, sys
log_prefix = '[' + os.path.basename(__file__)
class GP:
    # System Parameters
    n_episode = 1
    n_time_steps = 10
    delta_t   = 1
    @staticmethod
    def RESET(env, agent):
        env.reset()
        agent.reset()

    # System Model
    n_UEs = 20
    n_VM = 8 # number of VMs
    n_NF_inst = [2,2,2,2,2,2,1] # number of instances of AMF, SMF, UPF, UDM, UDR, AUSF
    nf_name   = ["AMF", "SMF", "UPF", "UDM", "UDR", "AUSF", "RISE"]
    next_nf   = [[["SMF",1],["AUSF",5]], [["AMF",0], ["UPF",2]], [["SMF",1]], [["UDR",4]], [["",-1]], [["UDM",3]],[["AMF", 0]]]
    req_type  = ["RegistrationRequest","AuthenticationResponse","SecurityModeComplete","IdentityResponse","RegistrationComplete","PDUSessionEstablishmentRequest"]

    # rate in NF/VM
    mu_RISE = 100

    # Log setting
    logDebugAvai = True
    logDataAvai  = False
    logOptional  = True
    logTopology  = True
    logRequest   = True
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

