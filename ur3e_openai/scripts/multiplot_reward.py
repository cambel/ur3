import glob
import numpy as np
from plotter_utils import *

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
matplotlib.style.use('seaborn')

def cumulative_reward_per_step(data, scale_factor=True):
    reward_details = np.array(data[:,2])
    total_rewards = [0]
    print('?', scale_factor)
    for episode_rewards in reward_details:
        for rw in episode_rewards:
            r = rw[0] + rw[1]
            if rw[2] < 0:
                r += -300
            if rw[3] > 10 or (not scale_factor and rw[0] > 0.68):
                r += 500
            # - step
            # if scale_factor:
            r -= 0.7
            total_rewards.append(total_rewards[-1]+r)

    return np.array(total_rewards)

# def cumulative_reward_per_step(rewards, scale_factor=None):
#     total_rewards = [0]
#     if scale_factor is not None:
#         scale_factor=scale_factor[3:]*0.2
#         print(len(rewards), len(scale_factor))
#         print(scale_factor)
#     else:
#         scale_factor = np.ones_like(rewards)
#     eps_counter = 0
#     for eps_rw, scale_factor in zip(rewards, scale_factor):
#         for r in eps_rw:
#             r = r*(1/scale_factor)
#             # print(r, r*(1/scale_factor),scale_factor)
            
#             if r < -10:
#                 r = -300
#                 eps_counter += 1
#             if r > 10:
#                 r = 500
#                 eps_counter += 1

#             total_rewards.append(total_rewards[-1]+r+0.3)
#         # if eps_counter > 445:
#         #     break
#     return np.array(total_rewards)

def sum_reward_per_eps(rewards, cumulative=False):
    reward_per_eps = [0]
    total = 0
    for rw in rewards:
        if cumulative:
            if rw[-1] > -450 and rw[-1] < 450:
                total += np.sum(rw)
            if rw[-1] > 450:
                total += np.sum(rw[:-1]) + 10
            if rw[-1] < -450:
                total -= np.sum(rw[:-1]) - 10
            reward_per_eps.append(total)
        else:
            reward_per_eps.append(np.sum(rw))
    return reward_per_eps

def flatten(rewards):
    rewards_flattened = np.array([])
    for rw in rewards:
        rewards_flattened = np.append(rewards_flattened, rw)
    return rewards_flattened

def load_common_data(folder_names):
    combined_data = [load_data(folder) for folder in folder_names]
    return np.concatenate(combined_data)

def load_data(folder_name):
    filename = glob.glob(ROOT+folder_name+'/state_*')
    if len(filename) > 0:
        return np.load(filename[0], allow_pickle=True)
    raise ValueError("File not found: %s" % folder_name)

def single_plot_detailed(folders, label='', color='C0', cumulative=False):
    all_data = []
    min_len = 10e8 
    for f in folders:
        scale = True if not 'Old' in label else False
        print(">?", not 'Old' in label, label)
        data =  cumulative_reward_per_step(load_data(f), scale)
        # data =  cumulative_reward_per_step(load_data(f)[:,1])
        all_data.append(data)
        if len(data) < min_len:
            min_len = len(data)

    all_data = np.array(all_data)

    for i in range(len(all_data)):
        all_data[i] = all_data[i][:min_len].flatten()

    print(all_data[0].shape, all_data[1].shape)

    all_data = np.stack((all_data[0], all_data[1]))
    y = np.average(all_data, axis=0)
    y_std = np.std(all_data, axis=0)
    # x = np.linspace(0, len(y), len(y))
    x = np.linspace(0, 100000, len(y))

    plt.plot(x, y, color=color,label=label)
    plt.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=0.2, facecolor=color)
    # ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)

def print_successful_episodes(data):
    rewards = np.array(data[:,1])
    reward_details = np.array(data[:,2])
    successes = 0
    for episode_rewards in reward_details:
        if episode_rewards[-1][3] > 0:
            successes += 1
        elif episode_rewards[-1][0] > 0.39:
            successes += 1
    print("num of successes", successes, 'out of', len(reward_details), '%', round(successes/len(reward_details),2))

def single_plot(data, label='', color='C0', cumulative=False, debug=False):
    obs = np.array(data[:,0])
    rewards = np.array(data[:,1])
    reward_details = np.array(data[:,2])
    print(label)
    print_successful_episodes(data)

    if debug:
        print(data.shape)
        print(np.array(obs[1]).shape)
        print(np.array(rewards[1]).shape)
        print(np.array(reward_details[1]).shape)

    rewards = np.delete(rewards, [0,1])

    rgb = colors.colorConverter.to_rgb(color)

    if cumulative:
        # y = np.array(sum_reward_per_eps(rewards, cumulative=True))
        scale_factor = None
        if 'rw' in label:
            if 'uniform' in label:
                print('hoooooooooo')
                scale_factor = np.load('/home/cambel/trufus/uniform_rw.npy')
            if 'normal' in label:
                print('yeeeee')
                scale_factor = np.load('/home/cambel/trufus/normal_rw.npy')
        y = cumulative_reward_per_step(data, scale_factor)
        # y = cumulative_reward_per_step(data[:,1], scale_factor)
        # y = np.log(y)
        x = np.linspace(0, len(y), len(y))
        print(len(y))
    else:
        y = np.array(sum_reward_per_eps(rewards))
        x = np.linspace(0, len(y), len(y))
        plt.plot(x,y,color=color, alpha=0.1)
        y = numpy_ewma_vectorized(np.array(sum_reward_per_eps(rewards)), 10)
    plt.plot(x,y,color=color,label=label)
    # plt.ylim(-450,450)

def multi_plot(folder_names, labels=None, cumulative=False):
    if not labels:
        labels = folder_names
    _colors = ['C0','C1','C2','C3','C4','C5','C6','C7']
    for i, folder in enumerate(folder_names):
        if isinstance(folder, list):
            single_plot_detailed(folder, label=labels[i], color=_colors[i], cumulative=cumulative)
        else:
            single_plot(load_data(folder), label=labels[i], color=_colors[i], cumulative=cumulative)

# Single plot of one or multiple trainings
# data = load_common_data(['20220124T015650.362872_SAC_pall_ft06'])
# single_plot(data)

# Multiplot
ROOT = '/root/o2ac-ur/tf2rl/results/'
folder_names = [
### Main Comparison ###

# ['learning_curve/0_pih_m24_no_cl', 'learning_curve/2_pih_m24_no_cl',],
# ['learning_curve/0_pih_m24_lin_cl_uniform', 'learning_curve/2_pih_m24_lin_cl_uniform',],
# ['learning_curve/0_pih_m24_lin_cl_normal', 'learning_curve/2_pih_m24_lin_cl_normal',],
# ['learning_curve/0_pih_m24_pro_cl_uniform', 'learning_curve/2_pih_m24_pro_cl_uniform',],
# ['learning_curve/0_pih_m24_pro_cl_normal', 'learning_curve/2_pih_m24_pro_cl_normal',],

### Ablation Studies ###
## PID scheduling vs Normal PID
# ['pid_scheduling/pih_pid_scheduling4', 'pid_scheduling/pih_pid_scheduling7'], # Our Adp. CL DyRe
# ['pid_scheduling/pih_normal_pid','pid_scheduling/pih_normal_pid2',],

## New reward vs Old reward
# ['reward_types/pih_m24_old_reward','reward_types/pih_m24_old_reward2',]
   ["20230216T025910.934135_SAC_slicing"]
]

labels = ['2048', '1024', '4096', '8192']
labels = ['per_episode', 'every_100_steps', 'every_200_steps', 'every_500_steps']
labels = ['No Curriculum', 'Linear Curriculum UDR', 'Linear Curriculum GDR', 'Adp. Curriculum UDR', 'Adp. Curriculum GDR']
labels = ['New reward function', 'Old reward function']
labels = ['PD gains scheduling','Normal PD', ]
labels = None
multi_plot(folder_names, labels=labels, cumulative=True)

plt.xlabel('Time Steps', size=20)
plt.ylabel('Cumulative Reward', size=20)
plt.ticklabel_format(style=('sci'), scilimits=(-5,3), useMathText=True)
# plt.tick_params(axis='both', labelsize='larger', which='minor', labelcolor='r')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# def forward(values):
#     return values/10000
# plt.xscale( 'function' , functions=(forward,  forward))  # You can include one of these two
# plt.yscale( 'linear' )  # lines, or both, or neither.
# plt.legend(title='Batch size')
# plt.legend(title='Update frequency')
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.show()
