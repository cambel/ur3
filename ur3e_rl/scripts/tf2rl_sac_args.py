#!/usr/bin/env python
import signal
import sys
import numpy as np
from ur3e_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params
import ur_control.utils as utils
from tf2rl.experiments.cb_trainer import Trainer
from tf2rl.algos.sac import SAC
import rospy
from shutil import copyfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow logging disabled
# import timeit


np.set_printoptions(suppress=True)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':

    parser = Trainer.get_argument()
    parser.add_argument('-e', '--env_id', type=int, help='environment ID', default=None)
    parser.add_argument('-c', '--config-file', type=str, help='Path to config YAML file', default=None)
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=5e3)  # still don't know what it this for
    parser.set_defaults(max_steps=50000)  # 10000 for training 200 for evaluation
    parser.set_defaults(save_model_interval=5000)
    parser.set_defaults(test_interval=1e10)  # 1e4 for training 200 for evaluation
    parser.set_defaults(test_episodes=1)
    parser.set_defaults(normalize_obs=False)
    parser.set_defaults(auto_alpha=True)  # ON vs OFF
    parser.set_defaults(use_prioritized_rb=True)
    parser.set_defaults(lr=3e-3)  # 1e4, 1e3
    parser.set_defaults(update_interval=1)  # update every so often. 0 = Every episode

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('ur3e_tf2rl',
                    anonymous=True,
                    log_level=rospy.INFO)

    clear_gym_params('ur3e_gym')
    clear_gym_params('ur3e_force_control')

    args = parser.parse_args()

    if args.evaluate:
        args.n_warmup = 0
        args.max_steps = 150
        args.test_interval = 1
        args.test_episodes = 10

    p = utils.TextColors()
    p.error("GYM Environment:{} ".format(args.config_file))

    common_ros_param_path = load_ros_params(rospackage_name="ur3e_rl",
                                            rel_path_from_package_to_file="config",
                                            yaml_file_name='simulation/common.yaml')

    ros_param_path = load_ros_params(rospackage_name="ur3e_rl",
                                     rel_path_from_package_to_file="config",
                                     yaml_file_name=args.config_file)

    args.episode_max_steps = rospy.get_param("ur3e_gym/steps_per_episode", 200)

    env = load_environment(
        rospy.get_param('ur3e_gym/env_id'),
        max_episode_steps=args.episode_max_steps)
    actor_class = rospy.get_param("ur3e_gym/actor_class", "default")
    seed = rospy.get_param("ur3e_gym/rand_seed", 0)
    batch_size = rospy.get_param("ur3e_gym/batch_size", args.batch_size)
    lr = rospy.get_param("ur3e_gym/lr", args.lr)
    auto_alpha = rospy.get_param("ur3e_gym/auto_alpha", args.auto_alpha)
    update_interval = rospy.get_param("ur3e_gym/update_interval", args.update_interval)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.seed(seed)

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        actor_class=actor_class,
        batch_size=batch_size,
        n_warmup=args.n_warmup,
        auto_alpha=auto_alpha,
        lr=lr,
        update_interval=update_interval,
    )
    trainer = Trainer(policy, env, args, test_env=None, seed=seed)
    outdir = trainer._output_dir
    rospy.set_param('ur3e_gym/output_dir', outdir)
    log_ros_params(outdir)
    copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")
    copyfile(common_ros_param_path, outdir + "/common_ros_gym_env_params.yaml")

    start_time = rospy.get_time()
    trainer()
    duration = (rospy.get_time() - start_time)/60.
    trainer.logger.info("Total duration: %s min" % (duration))
