import re
import matplotlib.pyplot as plt
import numpy as np

type = ['no-delay-ddpg','delay-ddpg']
date = '-0722'

color = ['red', 'green', 'blue']

pcolor= ['paleturquoise', 'peachpuff', 'y']

episode_reward = [[] for _ in range(len(type))]
complete_rate  = [[] for _ in range(len(type))]

for i in range(len(type)):
    file = 'data/procedure-'+type[i]+date+'.log'
    print(file)
    with open(file) as f:
        for line in f:
            ret = re.findall(r".*\[RODC-DDPG\]Episode-\d* reward: (\d*\.\d*)", line)
            if len(ret) > 0:
                episode_reward[i].append(float(ret[0]))
                print(ret[0])
            ret = re.findall(r".*\[RODC-DDPG\]Episode-\d*, complete procedure rate \((\d*) / (\d*) = (\d*\.\d*)\), average_service_time = (\d*\.\d*),.*", line)
            if len(ret) > 0:
                complete_rate[i].append([float(ret[0][0]), float(ret[0][1]), float(ret[0][2]), float(ret[0][3])])
                print(float(ret[0][0]), float(ret[0][1]), float(ret[0][2]), float(ret[0][3]))

fig, ax = plt.subplots()
for i in range(len(episode_reward)):
    ax.plot(episode_reward[i], color=color[i])
plt.savefig('data/episode-reward'+date+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()

fig, ax = plt.subplots()
for i in range(len(complete_rate)):
    complete_rate_np = np.array(complete_rate[i])
    evaluation_result = [a / b for a, b in zip(complete_rate_np[:,2], complete_rate_np[:,3])]
    print(len(evaluation_result))
    print(evaluation_result)
    ax.plot(evaluation_result, color=color[i])
plt.savefig('data/episode-complete'+date+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()