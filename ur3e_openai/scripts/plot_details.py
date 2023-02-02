import glob
import os
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
    rewards_per_eps = np.array(data[:, 2])
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


def load_detailed_data(data):
    """ load components of obs """
    obs_all = np.array(data[:, 0])
    print(obs_all.shape)
    eps_position_error = []
    eps_vel = []
    eps_last_action = []
    eps_force_torque = []

    for obs_t in obs_all:  # episodes
        obs_per_eps = np.array(obs_t)
        # print(obs_per_eps.shape)

        position_error = None
        vel = None
        last_action = None
        force_torque = None

        for obs in obs_per_eps:
            obs = obs[0][:-1].astype(np.float32)
            # print(obs.shape)
            # print(obs)
            if position_error is not None:
                # print('>>>')
                # print('>>>', position_error.shape, obs[:6].shape)
                position_error = np.vstack((position_error, obs[:6]))
                vel = np.vstack((vel, obs[6:12]))
                last_action = np.vstack((last_action, obs[12:36]))
                force_torque = np.vstack((force_torque, obs[36:]))
            else:
                # print('<<<')
                # print('<<<', obs)
                position_error = obs[:6]
                vel = obs[6:12]
                last_action = obs[12:36]
                force_torque = obs[36:]
            # print(position_error.shape)

        # print('f',position_error.shape)
        eps_position_error.append(position_error)
        eps_vel.append(vel)
        eps_last_action.append(last_action)
        eps_force_torque.append(force_torque)

    return (eps_position_error), \
           (eps_vel), \
           (eps_last_action), \
           (eps_force_torque)


def plot_details(data, save_only=False, folder=None, orientation=False):
    """ By DONE reward alone """
    num_eps = len(data[:, 0])
    eps_position_error, eps_vel, eps_last_action, eps_force_torque = load_detailed_data(data)
    eps_force_torque = prepare_force(eps_force_torque)
    eps_last_action = prepare_actions(eps_last_action)
    print('la', len(eps_last_action[0]), type(eps_last_action[0]))
    for i in range(len(eps_last_action[0])):
        for j in range(6):  
            eps_last_action[0][i][:, j] = numpy_ewma_vectorized(eps_last_action[0][i][:, j], 10)
            eps_last_action[1][i][:, j] = numpy_ewma_vectorized(eps_last_action[1][i][:, j], 10)
            eps_last_action[2][i][:, j] = numpy_ewma_vectorized(eps_last_action[2][i][:, j], 10)
            eps_last_action[3][i][:, j] = numpy_ewma_vectorized(eps_last_action[3][i][:, j], 10)
    # print('f', len(eps_force_torque))

    max_distance = np.array([0.04, 0.04, 0.05, 0.785398, 0.785398, 1.5707])
    max_force_torque = np.array([30.0, 30.0, 30.0, 4, 4, 4])
    max_velocity = np.array([0.5, 0.5, 0.5, 1.5707, 1.5707, 1.5707])
    goal_threshold = 0.0  # 05  # pulley

    for i in range(num_eps):
        time_steps = int(len(eps_position_error[i])*0.05)
        print('ts', time_steps)
        eps_position_error[i][:, 2] -= goal_threshold/0.05
    
        dist = eps_position_error[i] * max_distance * 1000
        plot_xyz(i, dist, legend='Translation Error (mm)',
                 orientation=orientation,
                 ylimits=[-40, 40], save_only=save_only, folder=folder,
                 ylimits2=[-4, 4], legend2='Orientation Error (rad)',
                 goal_threshold=True)
        plt.close()
        # plot_xyz(i, eps_vel, legend='Velocity (m/s)', orientation=orientation,
        #          ylimits=[-0.04, 0.04], save_only=save_only, folder=folder)
        force = eps_force_torque[i] * -1 * max_force_torque
        plot_xyz(i, force, legend='Force (N)',
                 orientation=orientation,
                 ylimits=[-15, 40], save_only=save_only, folder=folder,
                 ylimits2=[-0.372, 1], legend2='Torque (N m)')
        plt.close()


        last_action = eps_last_action[0]
        plot_xyz(i, last_action[i], legend='Action Motion (Normalized)',
                 orientation=orientation, ylimits=[-1, 1], ylimits2=[-1, 1],
                 save_only=save_only, folder=folder)
        plt.close()
        
        last_action = eps_last_action[1]
        plot_xyz(i, last_action[i], legend='Action Position PD (Normalized)',
                 orientation=orientation, ylimits=[-1, 1], ylimits2=[-1, 1],
                 save_only=save_only, folder=folder)
        plt.close()
        
        last_action = eps_last_action[2]
        plot_xyz(i, last_action[i], legend='Action Force PI (Normalized)',
                 orientation=orientation, ylimits=[-1, 1], ylimits2=[-1, 1],
                 save_only=save_only, folder=folder)
        plt.close()

        last_action = eps_last_action[3]
        plot_xyz(i, last_action[i], legend='Action Selection Matrix (Normalized)',
                 orientation=orientation, ylimits=[-1, 1], ylimits2=[-1, 1],
                 save_only=save_only, folder=folder)


def plot_dist_force_z_comparison(
        idx, data1, data2, label1, label2, group='', save_only=False, colors=('tab:blue', 'tab:orange')):

    dist1, _, _, force1 = load_detailed_data(data1)
    dist2, _, _, force2 = load_detailed_data(data2)

    fig, ax1 = plt.subplots()

    max_distance = np.array([0.04, 0.04, 0.05, 0.785398, 0.785398, 1.5707])
    max_force_torque = np.array([30, 30.0, 30.0, 4, 4, 4])

    d1 = dist1[idx] * max_distance * 1000
    f1 = prepare_force(force1)[idx] * max_force_torque * -1 #* 0.6
    twin_plot(ax1, 2, d1, f1, label=label1, ls='-', alpha=1.0, colors=colors)

    d2 = dist2[idx] * max_distance * 1000
    f2 = prepare_force(force2)[idx] * max_force_torque * -1
    twin_plot(ax1, 2, d2, f2, label=label2, ls='-.', alpha=0.5, colors=colors)

    plt.axhline(0, color='black', ls=':', alpha=0.3)
    plt.axhline(2, color=colors[1], ls=':', alpha=0.5, label='insertion threshold')
    fig.tight_layout()

    if save_only:
        filename = str(i)+'comp_dist_force_' + group
        plt.savefig('/home/cambel/Pictures/'+filename)
        plt.clf()
    else:
        plt.show()


def prepare_actions(data):
    actions_x = []
    actions_pd = []
    actions_pi = []
    actions_alpha = []
    for y in data:
        y = y.reshape(y.shape[0], 6, -1)
        actions_x.append(y[:, :, 0])
        actions_pd.append(y[:, :, 1])
        actions_pi.append(y[:, :, 2])
        actions_alpha.append(y[:, :, 3])
    return actions_x, actions_pd, actions_pi, actions_alpha


def prepare_force(data):
    result = []
    for y in data:
        y = y.reshape(y.shape[0], 6, -1)
        y = np.average(y, axis=1)
        result.append(y)
    return result


def prepare_pose_error(data):
    pass


def twin_plot(ax1, axis, data1, data2, label, ls='-', alpha=1.0, colors=('tab:blue', 'tab:orange')):
    """ TBD
    """
    y1 = data1
    y2 = data2

    ax1.set_xlabel('Time (s)', fontsize='x-large')
    ax1.set_ylabel('Distance error (mm)', color=colors[0], fontsize='x-large')

    time_steps = int(len(y1)*0.05)
    # print('ts', time_steps)
    x1 = np.linspace(0, time_steps, len(y1))
    ax1.plot(x1, y1[:, axis], label=label, ls=ls, color=colors[0], alpha=alpha)
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.set_ylim(bottom=0, top=50)
    ax1.legend(fontsize='large')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (N)', color=colors[1], fontsize='x-large')  # we already handled the x-label with ax1
    x2 = np.linspace(0, time_steps, len(y2))
    ax2.plot(x2, y2[:, axis], label=label, ls=ls, color=colors[1], alpha=alpha)
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_ylim(bottom=-1, top=30)
    ax1.legend(fontsize='large')


def plot_xyz(idx, variable_data, legend='', orientation=False, 
             xlimits=None, ylimits=None, save_only=False, folder='',
             ylimits2=None, legend2='', goal_threshold=False):
    """ orientation False => Position related variables 
        orientation True => Orientation related variables
    """
    y = variable_data
    # print('y', y.shape)

    time_steps = int(len(y)*0.05)
    # print('ts', time_steps)
    x = np.linspace(0, time_steps, len(y))
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize='x-large')
    ax1.set_ylabel(legend, color='black', fontsize='x-large')


    ax1.plot(x, y[:, 0], label='x', c='b', alpha=0.8)
    ax1.plot(x, y[:, 1], label='y', c='g', alpha=0.8)
    ax1.plot(x, y[:, 2], label='z', c='r', alpha=0.8)
    
    if goal_threshold:
        ax1.axhline(5, color='r', ls=':', alpha=0.5, label='insertion threshold')

    ax2 = ax1.twinx()
    ax2.set_ylabel(legend2, color='gray', alpha=0.6, fontsize='x-large')  # we already handled the x-label with ax1

    if orientation:
        orientation_label = "a" 
        ax2.plot(x, y[:, 3], label=orientation_label+'x', c='b', ls='--', alpha=0.5)
        ax2.plot(x, y[:, 4], label=orientation_label+'y', c='g', ls='--', alpha=0.5)
        ax2.plot(x, y[:, 5], label=orientation_label+'z', c='r', ls='--', alpha=0.5)

    plt.axhline(0, color='black', ls=':', alpha=0.3)
    ax1.set_xlim(xlimits)
    ax1.set_ylim(ylimits)
    ax2.set_ylim(ylimits2)
    ax2.tick_params(axis='y', labelcolor='gray', grid_alpha=0.6)

    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.xlabel('Time (s)', size='x-large')
    # plt.ylabel(legend, size='x-large')
    fig.tight_layout()

    if save_only:
        if not os.path.exists(folder):
            os.makedirs(folder)

        if orientation:
            filename = str(idx)+'_o_'+legend.lower().replace(' ', '_')
        else:
            filename = str(idx)+'_'+legend.lower().replace(' ', '_')
        plt.savefig(folder+'/'+filename)
        plt.clf()
        return
    plt.show()


def get_avg_steps(data):
    obs_per_eps = np.array(data[:, 0])
    # print(np.array(obs_per_eps[1]).shape)


def load_data(folder_names, part='pulley'):
    data = None
    for folder in folder_names:
        filename = glob.glob('/home/cambel/dev/results/'+folder+'/*%s*' % part)
        # print(filename)
        if data is not None:
            data = np.concatenate([data, np.load(filename[0], allow_pickle=True)])
        else:
            data = np.load(filename[0], allow_pickle=True)
    return data

###########################################################
#  Plot details
###########################################################

data = np.load(
    # NO CL
    # '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T021157.npy',
    # '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T053808-shaft30.npy',
    '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T063135-bearing.npy',
    # GRD WR
    # '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T040156-pulley.npy',
    # '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T051349-shaft.npy',
    # '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T061526-bearing.npy',

    allow_pickle=True)

plot_details(data, orientation=True, folder='bearing_nocl', save_only=True)
###########################################################

###########################################################
####### Plot Z-axis distance from target and force ########
###########################################################

# colors = ('tab:blue', 'tab:orange')
# colors = ('#DBB40C', 'tab:purple')
# colors = ('tab:green', 'tab:red')

# # data1 = np.load('/home/cambel/dev/results/3-20/3-20_pih_m24_pro_cl_normal_rw/state_20220325T015510-pulley.npy', allow_pickle=True)
# data1 = np.load('/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T040156-pulley.npy', allow_pickle=True)
# # data1 = np.load(
# #     '/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T051349-shaft.npy',
# #     allow_pickle=True)
# # data1 = np.load('/home/cambel/dev/results/3-20/newmouse_1_pih_m24_pro_cl_normal_rw/state_20220325T061526-bearing.npy', allow_pickle=True)

# data2 = np.load('/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T021157.npy', allow_pickle=True)
# # data2 = np.load('/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T053808-shaft30.npy', allow_pickle=True)
# # data2 = np.load('/home/cambel/dev/results/3-20/newmouse_1_pih_m24_no_cl/state_20220325T063135-bearing.npy', allow_pickle=True)

# # plot_dist_force_z_comparison(8, data1, data2, 'GDR DyRe', 'No CL')
# for i in range(30):
#     plot_dist_force_z_comparison(i, data1, data2, 'Adp. Curriculum GDR DyRe', 'No Curriculum', 
#                                     colors=colors, group='pulley', save_only=True)

###########################################################


# TODO: Get indexes of success >
#         used them on detailed data to get the initial distance & number of steps >
#           normalize number of steps


# TODO: Plot grouped reward data
