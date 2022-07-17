from environment import ENV
from agent import Agent
from globalParameters import GP
from scipy import stats

import os,sys
log_prefix = '[' + os.path.basename(__file__)

if __name__ == '__main__':
    GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Line-1][Initialize env and agent]',None,'debug')
    env, agent = ENV(), Agent()
    for ep in range(GP.n_episode):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-9][Training Episode - %d]',ep+1,'debug')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-10][Reset env and agent]', None, 'debug')
        GP.RESET(env,agent)
        UeReqs = stats.poisson.rvs(mu=GP.n_UEs, size=GP.n_time_steps+10, random_state=None)
        env.episode_reward.append(0)
        for ts in range(GP.n_time_steps):
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-11][Training Time Steps - %d]',ts,'debug')
            #env.update_ue_reqs_every_time_step(UeReqs[ts], ts)
            env.update_ue_reqs_every_time_step(30,ts)
            obs_reward = env.send_obs_reward(ts)
            GP.obs_on_road.append(obs_reward)
            action = agent.receive_observations(GP.CHECK_OBSERVATIONS(ts),ts)
            env.execute_action(action)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'Episode-%d reward: %f', (ep, env.episode_reward[-1]), 'data')