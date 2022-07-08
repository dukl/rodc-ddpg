from environment import ENV
from agent import Agent
from globalParameters import GP

import os
log_prefix = '[' + os.path.basename(__file__) +'][RODC-DDPG]'

if __name__ == '__main__':
    GP.LOG(log_prefix+'[Line-1][Initialize env and agent]',None,'debug')
    env, agent = ENV(), Agent()
    for ep in range(GP.n_episode):
        GP.LOG(log_prefix+'[line-9][Training Episode - %d]',ep+1,'debug')
        GP.LOG(log_prefix+'[line-10][Reset env and agent]', None, 'debug')
        GP.RESET(env,agent)
        for ts in range(GP.n_time_steps):
            GP.LOG(log_prefix+'[line-11][Training Time Steps - %d]',ts,'debug')
            obs_reward = env.send_obs_reward(ts)
            GP.obs_on_road.append(obs_reward)
            action = agent.receive_observations(GP.CHECK_OBSERVATIONS(ts),ts)
            obs_reward = env.execute_action(action)