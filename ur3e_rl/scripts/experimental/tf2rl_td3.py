#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #tensorflow logging disabled

import rospy
import timeit
import argparse
from gym.envs.registration import register

from tf2rl.algos.td3 import TD3
from tf2rl.experiments.trainer import Trainer

import ur_control.utils as utils

from ur3e_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params
import numpy as np
np.set_printoptions(suppress=True)

import sys
import signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':

    parser = Trainer.get_argument()
    parser.add_argument('--env-id', type=int, help='environment ID', default=None)
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=0) # still don't know what it this for
    parser.set_defaults(max_steps=150000) # 10000 for training 200 for evaluation
    parser.set_defaults(save_model_interval=1000)
    parser.set_defaults(test_interval=16e5) # 1e4 for training 200 for evaluation
    parser.set_defaults(test_episodes=1)
    parser.set_defaults(normalize_obs=False)
    parser.set_defaults(auto_alpha=True)
    parser.set_defaults(use_prioritized_rb=True)
    parser.set_defaults(lr=3e-4)

    args = parser.parse_args(rospy.myargv()[1:])


    rospy.init_node('ur3e_tf2rl',
                    anonymous=True,
                    log_level=rospy.ERROR)

    clear_gym_params('ur3e_gym')

    start_time = timeit.default_timer()

    args = parser.parse_args()
    param_file = None
    
    if args.evaluate:
        args.n_warmup = 0
        args.max_steps = 150
        args.test_interval = 1
        args.test_episodes = 10

    if args.env_id == 0:
        args.dir_suffix = "hybrid_sim_9"
        param_file = "simulation/hybrid9.yaml"
    elif args.env_id == 1:
        args.dir_suffix = "hybrid_sim_14"
        param_file = "simulation/positive_cost/hybrid14.yaml"
    elif args.env_id == 2:
        args.dir_suffix = "hybrid_sim_19"
        param_file = "simulation/hybrid19.yaml"
    elif args.env_id == 3:
        args.dir_suffix = "hybrid_sim_24"
        param_file = "simulation/hybrid24.yaml"
    elif args.env_id == 4:
        args.dir_suffix = "impedance_sim8"
        param_file = "simulation/impedance8.yaml"
    elif args.env_id == 5:
        args.dir_suffix = "impedance_sim13pd"
        param_file = "simulation/positive_cost/impedance13pd.yaml"
    elif args.env_id == 6:
        args.dir_suffix = "impedance_sim13imp"
        param_file = "simulation/impedance13.yaml"
    elif args.env_id == 7:
        args.dir_suffix = "impedance_sim18"
        param_file = "simulation/positive_cost/impedance18.yaml"
    elif args.env_id == 8:
        args.dir_suffix = "gears_hybrid"
        param_file = "real/hybrid_gears.yaml"
    elif args.env_id == 9:
        args.dir_suffix = "gears_imp"
        param_file = "real/imp_gears.yaml"
    elif args.env_id == 10:
        args.dir_suffix = "wood_toy_hybrid"
        param_file = "real/hybrid_wood_toy.yaml"
    elif args.env_id == 11:
        args.dir_suffix = "wood_toy_imp"
        param_file = "real/imp_wood_toy.yaml"
    elif args.env_id == 12:
        args.dir_suffix = "fujifilm"
        param_file = "simulation/fujifilm.yaml"
    elif args.env_id == 13:
        args.dir_suffix = "fujifilmI"
        param_file = "simulation/fujifilmI-13.yaml"
    else:
        raise Exception("invalid env_id")

    p = utils.TextColors()
    p.error("GYM Environment:{} ".format(param_file))

    load_ros_params(rospackage_name="ur3e_rl",
                    rel_path_from_package_to_file="config",
                    yaml_file_name=param_file)

    args.episode_max_steps = rospy.get_param("ur3e_gym/steps_per_episode", 100)

    env = load_environment(
            rospy.get_param('ur3e_gym/env_id'),
            max_episode_steps=args.episode_max_steps)

    policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        # gpu=args.gpu,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup
        )
    trainer = Trainer(policy, env, args, test_env=None)
    outdir = trainer._output_dir
    log_ros_params(outdir)
    trainer()

    print("duration", (timeit.default_timer() - start_time)/60.,"min")

# rosrun ur3e_rl tf2rl_sac.py --env-id=0