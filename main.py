from environment import ENV
from agent import Agent
from globalParameters import GP

if __name__ == '__main__':
    GP.LOG('[System][RODC-DDPG][Line-1][Initialize env and agent]',None,'debug')
    env, agent = ENV(), Agent()
    for ep in range(GP.n_episode):
        GP.LOG('[System][RODC-DDPG][line-9][Training Episode - %d]',ep+1,'debug')
        GP.LOG('[System][RODC-DDPG][line-10][Reset env and agent]', None, 'debug')
        GP.RESET(env,agent)
        for ts in range(GP.n_time_steps):
            GP.LOG('[System][RODC-DDPG][line-11][Training Time Steps - %d]',ts,'debug')
            action = agent.receive_observations(GP.CHECK_OBSERVATIONS(ts),ts)
