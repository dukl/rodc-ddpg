from logger import log_debug,log_data
class GP:
    # System Parameters
    n_episode = 2
    n_time_steps = 2
    delta_t   = 1
    @staticmethod
    def RESET(env, agent):
        env.reset()
        agent.reset()

    # agent parameters
    sc = None # latest observation RODC-DDPG Line-6

    # Log setting
    logDebugAvai = True
    logDataAvai  = True
    logOptional  = True
    @staticmethod
    def LOG(str, value, type):
        if GP.logDebugAvai and type is 'debug':
            if value is not None:
                log_debug.logger.debug(str,value)
            else:
                log_debug.logger.debug(str)
        if GP.logDataAvai and type is 'data':
            if value is not None:
                log_data.logger.debug(str,value)
            else:
                log_data.logger.debug(str)
        if GP.logOptional and type is 'optional':
            if value is not None:
                log_debug.logger.debug(str,value)
            else:
                log_debug.logger.debug(str)

    # System Running
    obs_on_road = []
    @staticmethod
    def CHECK_OBSERVATIONS(ts):
        O= []
        for ob in GP.obs_on_road:
            if ob.id + ob.n_ts <= ts:
                O.append(ob)
                GP.obs_on_road.remove(ob)
        GP.LOG('[System][RODC-DDPG][line-12][received observations (n = %d)]',len(O),'optional')

