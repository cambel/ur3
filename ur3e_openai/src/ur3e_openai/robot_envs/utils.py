import rospy
import numpy as np
from ur_gazebo.model import Model
from ur_gazebo.basic_models import PEG_BOARD, BOX
from ur_control import transformations as tr, conversions
from matplotlib import cm

def load_param_vars(self, prefix):
    """
        Dynamically load class variables from ros params based on a prefix
    """
    params = rospy.get_param_names()
    for param in params:
        if prefix in param:
            var_name = '_'.join(param.split('/')[2:])
            value = rospy.get_param(param)
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, var_name, value)

def verify_reward_episode(reward_per_step):
    goal_index = 0
    for i, step in reward_per_step:
        if step[5] >= 0:
            goal_index = i
            break
    return reward_per_step[:goal_index]

def save_log(obs_logfile, obs_per_step, reward_per_step, cost_ws):
    if len(obs_per_step) == 0:
        return

    if len(reward_per_step) > 0:
        reward_per_step = np.array(reward_per_step)
        reward_per_step = np.sum(reward_per_step, axis=0).flatten()

        preward = np.copy(reward_per_step)
        ws = (cost_ws/cost_ws.sum()) if cost_ws.sum() > 0 else cost_ws
        preward[:3] = preward[:3] * ws
        print("d: %s, f: %s, a: %s, r: %s, cs: %s, ik: %s, cll: %s" % tuple(np.round(preward.tolist(), 1)))

    try:
        tmp = np.load(obs_logfile, allow_pickle=True).tolist()
        tmp.append(obs_per_step)
        np.save(obs_logfile, tmp)
        tmp = None
    except IOError:
        np.save(obs_logfile, [obs_per_step], allow_pickle=True)


def apply_workspace_contraints(target_pose, workspace):
    """
        target pose: Array. [x, y, z, ax, ay, az]
        workspace: Array. [[min_x, max_x], [min_y, max_y], ..., [min_az, max_az]]

        Even if the action space is smaller, define the workspace for every dimension
    """
    if len(target_pose) == 7:
        tpose = np.concatenate([target_pose[:3], tr.euler_from_quaternion(target_pose[3:])])
        res = np.array([np.clip(target_pose[i], *workspace[i]) for i in range(6)])
        return np.concatenate([res[:3], tr.quaternion_from_euler(*res[3:])])
    else:
        return np.array([np.clip(target_pose[i], *workspace[i]) for i in range(6)])

def randomize_pose(center_pose, workspace, reset_time, rng):
    """
    Given a cartesian pose and a workspace, define a random translation/rotation around the given pose.
    """
    rand = rng.uniform(low=-1.0, high=1.0, size=6)
    rand = np.array([np.interp(rand[i], [-1., 1.], workspace[i]) for i in range(6)])
    rand[3:] = np.deg2rad(rand[3:]) / reset_time  # rough estimation of angular velocity
    return tr.transform_pose(center_pose, rand)

def simple_random(workspace, rng):
    options = []
    cbw = workspace
    options.append([cbw[0][0], cbw[1][0], 0, 0, 0, 0])
    options.append([cbw[0][0], cbw[1][1], 0, 0, 0, 0])
    options.append([cbw[0][1], cbw[1][0], 0, 0, 0, 0])
    options.append([cbw[0][1], cbw[1][1], 0, 0, 0, 0])
    options.append([cbw[0][0], cbw[1][0], 0, 0, 0, cbw[5][0]])
    options.append([cbw[0][0], cbw[1][1], 0, 0, 0, cbw[5][0]])
    options.append([cbw[0][1], cbw[1][0], 0, 0, 0, cbw[5][0]])
    options.append([cbw[0][1], cbw[1][1], 0, 0, 0, cbw[5][0]])
    rand = rng.choice(options)
    rand[3:] = np.deg2rad(rand[3:])
    return rand

def create_gazebo_marker(pose, reference_frame, marker_id=None):
    marker_pose = [pose[:3].tolist(), pose[3:].tolist()]
    return Model("visual_marker", marker_pose[0], orientation=marker_pose[1], reference_frame=reference_frame, model_id=marker_id)


def create_box(box_pose):
    obj = BOX % ("box", "0.1", "0.1", "0.01", "Black", "0.1", "0.1", "0.01")
    model_names = ["box"]
    objpose = [box_pose[:3].tolist(), box_pose[3:].tolist()]
    return Model(model_names[0], objpose[0], orientation=objpose[1], file_type='string', string_model=obj, reference_frame="base_link")


def create_peg_board(board_pose, gazebo_params):
    stiffness = 0.0
    stiff_upper = gazebo_params['stiff_upper_limit']  # 2e5
    stiff_lower = gazebo_params['stiff_lower_limit']  # 5e4
    if gazebo_params['stiffness'] == 'random':
        stiffness = np.random.randint(stiff_lower, stiff_upper)
    else:
        stiffness = gazebo_params['stiffness']
    r, g, b = get_board_color(stiffness, stiff_upper, stiff_lower)
    transparency = 0.8
    name = "peg_board"
    objpose = [board_pose[:3].tolist(), board_pose[3:].tolist()]

    string_model = PEG_BOARD.format(transparency, r, g, b, stiffness)
    return Model(name, objpose[0], orientation=objpose[1], file_type='string', string_model=string_model, reference_frame="base_link")


def peg_in_hole_models(board_pose, gazebo_params, marker_poses=None):
    peg_board = create_peg_board(board_pose, gazebo_params)
    bottom_pose = np.array(conversions.transform_end_effector(board_pose, [0., 0., -0.015, 0, 1, 0, 0]))
    bottom_box = create_box(bottom_pose)
    markers = []
    if marker_poses is not None:
        assert isinstance(marker_poses, list)
        cnt = 1
        for mp in marker_poses:
            markers.append(create_gazebo_marker(mp, "base_link", marker_id="marker%s" % cnt))
            cnt += 1
    models = [peg_board]
    models += [bottom_box]
    models += markers
    return models


def get_board_color(stiffness, stiff_upper, stiff_lower):
    value = np.interp(stiffness, [stiff_lower, stiff_upper], [0, 1])
    viridis = cm.get_cmap('viridis', 10)
    return viridis(value)

def get_value_from_range(value, base, range_constant, mtype='sum'):
    if mtype == 'sum':
        kp_min = base - range_constant
        kp_max = base + range_constant
        return np.interp(value, [-1, 1], [kp_min, kp_max])
    elif mtype == 'mult':
        kp_min = base / range_constant
        kp_max = base * range_constant
        if value >= 0:
            return np.interp(value, [0, 1], [base, kp_max])
        else:
            return np.interp(value, [-1, 0], [kp_min, base])


def concat_vec(x, y, z, steps):
    x = x.reshape(-1, steps)
    y = y.reshape(-1, steps)
    z = z.reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T


def spiral(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y


def get_conical_helix_trajectory(p1, p2, steps, revolutions=5.0):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist, 0, steps)
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)
