from environment import ENV
from agent import Agent
from logger import log

if __name__ == '__main__':
    log.logger.debug('[System][RODC-DDPG][Line-1][Initialize env and agent]')
    env, agent = ENV(), Agent() 
