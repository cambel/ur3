import glob
import numpy as np
from plotter_utils import numpy_ewma_vectorized, smooth

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def sum_reward_per_eps(rewards):
    reward_per_eps = []
    for rw in rewards:
        reward_per_eps.append(np.sum(rw))
    return reward_per_eps

def flatten(rewards):
    rewards_flattened = np.array([])
    for rw in rewards:
        rewards_flattened = np.append(rewards_flattened, rw)
    return rewards_flattened

def load_data(folder_names):
    data = None
    for folder in folder_names:
        filename = glob.glob('/home/cambel/dev/results/'+folder+'/state_*')
        print(filename)
        if data is not None:
            data = np.concatenate([data, np.load(filename[0], allow_pickle=True)])
        else:
            data = np.load(filename[0], allow_pickle=True)
    return data

data = load_data(['3-20/newmouse_1_pih_m24_no_cl'])

# episode: ( obs:(position[6], velocity[6], force[6], action_result[1]), 
#            reward[1] )
obs = np.array(data[:,0])
rewards = np.array(data[:,1])
reward_details = np.array(data[:,2])

print(data.shape)

print(np.array(obs[1]).shape)
print(np.array(rewards[1]).shape)
print(np.array(reward_details[1]).shape)

# rewards = np.delete(rewards, [0,1])

# y = np.array(reward_details[-1])[:,-1]
# print("rf", y)

y = np.array(sum_reward_per_eps(rewards))
x = np.linspace(0, len(y), len(y))
plt.plot(x,y,color=[0.5,0.5,0.5,0.5])
y = numpy_ewma_vectorized(np.array(sum_reward_per_eps(rewards)), 5)
plt.plot(x,y)
# plt.ylim(-450,450)
plt.show()