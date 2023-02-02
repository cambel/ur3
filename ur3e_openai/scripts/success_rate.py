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

def get_success_rate(data, position_threshold=0.8893):
    """ By DONE reward alone """
    rewards_per_eps = np.array(data[:,2])
    # print(np.array(rewards_per_eps[1]).shape)
    success_count = 0
    steps_count = 0
    for rw in rewards_per_eps:
        if rw[-1][-2] > 0:
            success_count += 1
            steps_count += len(rw)
    if position_threshold:
        success_count2 = 0
        steps_count2 = 0
        for rw in rewards_per_eps:
            if rw[-1][0] > position_threshold:
                success_count2 += 1
                for r in rw:
                    steps_count2 += 1
                    if r[0] > position_threshold:
                        break
        print('extra', success_count2, steps_count2/success_count2+0.01)
    return success_count, len(rewards_per_eps), steps_count/(success_count+0.01)

def get_success_rate_obs(data):
    """ By DONE reward alone """
    obs_per_eps = np.array(data[:,0])
    success_count = 0
    print(np.array(obs_per_eps[1]).shape)
    steps_count = 0
    for rw in obs_per_eps:
        if rw[-1][-2] > 0:
            success_count += 1
            steps_count += len(rw)
    return success_count, len(obs_per_eps), steps_count/(success_count+0.01)

def get_avg_steps(data):
    obs_per_eps = np.array(data[:,0])
    # print(np.array(obs_per_eps[1]).shape)
    
def load_data(folder_names, part='pulley'):
    data = None
    for folder in folder_names:
        filename = glob.glob('/home/cambel/dev/results/'+folder+'/*%s*'%part)
        # print(filename)
        if data is not None:
            data = np.concatenate([data, np.load(filename[0], allow_pickle=True)])
        else:
            data = np.load(filename[0], allow_pickle=True)
    return data

def group_results(folders, part=None):
    print('Results for;', part)
    for folder in folders:
        data = load_data([folder], part)
        print(folder, 'success rate', get_success_rate(data))


# group_results(
#     [
#         # 'mouse/newmouse_1_pih_m24_no_cl',
#         # 'mouse/newmouse_1_pih_m24_pro_cl_normal', 
#         'mouse/newmouse_1_pih_m24_pro_cl_normal_rw', 
#         ],
#         'bearing')

# data = load_data(['mouse/newmouse_1_pih_m24_pro_cl_normal'])
# data = np.load('/home/cambel/dev/results/mouse/newmouse_1_pih_m24_pro_cl_normal/state_20220308T063810-shaft.npy', allow_pickle=True)
# data = np.load('/home/cambel/dev/results/mouse/newmouse_1_pih_m24_pro_cl_normal/state_20220308T130034-start.npy', allow_pickle=True)

# episode: ( obs:(position[6], velocity[6], force[6], action_result[1]), 
#            reward[1] )
# data = np.load('/home/cambel/dev/results/mouse/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220308T081516-bearing.npy', allow_pickle=True)
# reward_details = np.array(data[:,2])
# maxr = 1000
# for r in reward_details:
#     if r[-1][-2] > 0 and r[-1][0] < maxr:
#         maxr = r[-1][0]
# print('pos threshold rw', maxr)
# print(np.array(reward_details[3])[-1].tolist())
# obs = np.array(data[:,0])
# rewards = np.array(data[:,1])

# print(data.shape)

# print(np.array(obs[2]).shape)
# print(np.array(rewards[1]).shape)
# print(np.array(reward_details[-1]).shape)

# print('success rate', get_success_rate(data))
# print('svg steps', get_avg_steps(data))
# rewards = np.delete(rewards, [0,1])

# y = np.array(reward_details[-1])[:,-1]
# print("rf", y)

# y = np.array(sum_reward_per_eps(rewards))
# x = np.linspace(0, len(y), len(y))
# plt.plot(x,y,color=[0.5,0.5,0.5,0.5])
# y = numpy_ewma_vectorized(np.array(sum_reward_per_eps(rewards)), 5)
# plt.plot(x,y)
# # plt.ylim(-450,450)
# plt.show()