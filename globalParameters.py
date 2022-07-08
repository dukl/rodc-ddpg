from logger import log_debug,log_data
class GP:
    # System Parameters
    n_episode = 2
    n_time_steps = 2

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