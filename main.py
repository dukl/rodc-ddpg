from environment import ENV
from agent import Agent
from globalParameters import GP
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import os,sys
log_prefix = '[' + os.path.basename(__file__)

if __name__ == '__main__':
    GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Line-1][Initialize env and agent]',None,'debug')
    env, agent = ENV(), Agent()
    for ep in range(GP.n_episode):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-9][Training Episode - %d]',ep+1,'debug')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-10][Reset env and agent]', None, 'debug')
        GP.RESET(env,agent)
        #UeReqs = stats.poisson.rvs(mu=GP.n_UEs, size=GP.n_time_steps+10, random_state=None)
        #UeReqs, tmp = [GP.n_UEs], [0 for _ in range(GP.n_time_steps + 10)]
        #UeReqs.extend(tmp)
        env.episode_reward.append(0)
        for ts in range(GP.n_time_steps):
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-11][Training Time Steps - %d]',ts,'debug')
            #env.update_ue_reqs_every_time_step(UeReqs[ts], ts)
            env.update_ue_reqs_every_time_step(10,ts)
            obs_reward = env.send_obs_reward(ts)
            GP.obs_on_road.append(obs_reward)
            #action = agent.receive_observations(GP.CHECK_OBSERVATIONS(ts),ts)
            action = agent.receive_observation_no_delay(GP.CHECK_OBSERVATIONS(ts), ts)
            env.execute_action(action)
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'Episode-%d reward: %f', (ep, env.episode_reward[-1]), 'data')
        n_ue_succ, average_service_time = 0, 0
        for ue in env.ues:
            if ue.end_time != 0:
                n_ue_succ += 1
                average_service_time += ue.end_time - ue.start_time
        if n_ue_succ != 0:
            average_service_time = average_service_time/n_ue_succ
            GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'Episode-%d, complete procedure rate (%d / %d = %f), average_service_time = %f, evaluation_value = %f', (ep, n_ue_succ, len(env.ues), n_ue_succ/len(env.ues), average_service_time, (n_ue_succ/len(env.ues))/average_service_time), 'data')
        cpu_hist = []
        for nf in env.nfs:
            for inst in nf:
                cpu_hist.append(inst.history_cpu_per_ts)
        np_cpu_hist = np.array(cpu_hist)
        for cpu in np_cpu_hist:
            plt.plot([i for i in range(GP.n_time_steps)], cpu)
        #plt.show()
        plt.savefig('cpu dynamics '+str(GP.nf_cpu_change_rate_sin)+'.png')