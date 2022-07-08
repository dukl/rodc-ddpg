from logger import log_debug,log_data
import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'
class GP:
    # System Parameters
    n_episode = 1
    n_time_steps = 10
    delta_t   = 1
    @staticmethod
    def RESET(env, agent):
        env.reset()
        agent.reset()

    # Log setting
    logDebugAvai = True
    logDataAvai  = True
    logOptional  = True
    @staticmethod
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

    # System Running
    obs_on_road = []
    @staticmethod
    def CHECK_OBSERVATIONS(ts):
        O= []
        GP.LOG(log_prefix+'[line-7][Initialize the received observations O: len(O)=%d]',(len(O)), 'optional')
        for ob in GP.obs_on_road:
            if ob.id + ob.n_ts <= ts:
                O.append(ob)
                GP.obs_on_road.remove(ob)
        GP.LOG(log_prefix+'[line-13][received observations len(O)=%d]',len(O),'optional')
        for ob in O:
            GP.LOG(log_prefix+'[line-13][received observation s[%d]]', ob.id, 'optional')
        return O

