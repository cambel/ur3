#!/usr/bin/env python
import ur_control.utils as utils
import signal
import sys
import timeit
from tf2rl.algos.sac import SAC
from ur3e_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params
import rospy
import tensorflow as tf
import numpy as np
import argparse
import os

from ur3e_openai.initialize_logger import initialize_logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow logging disabled
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_env(folder, common_test, real):
    gyv_envs_params = folder + '/ros_gym_env_params.yaml'
    # gyv_envs_params = folder + '/common_ros_gym_env_params.yaml'

    assert os.path.exists(gyv_envs_params)

    rospy.set_param("ur3e_gym/rand_init_interval", 1)
    if real:
        load_ros_params(rospackage_name="ur3e_rl",
                        rel_path_from_package_to_file="config",
                        yaml_file_name="real/real_common.yaml")
        print("loading config file:", "real/real_common.yaml")
    else:
        load_ros_params(rospackage_name="ur3e_rl",
                        rel_path_from_package_to_file="config",
                        yaml_file_name=gyv_envs_params)
        print("loading config file:", gyv_envs_params)

    if common_test and not real:
        print("common params")
        load_ros_params(rospackage_name="ur3e_rl",
                        rel_path_from_package_to_file="config",
                        yaml_file_name='simulation/force_control/pih_test.yaml')
        print("loading config file:", 'simulation/force_control/pih_test.yaml')

    steps_per_episode = rospy.get_param("ur3e_gym/steps_per_episode", 200)

    return load_environment(rospy.get_param('ur3e_gym/env_id'),
                            max_episode_steps=steps_per_episode)


def test_policy(env, policy, num_tests, custom_path=False, training=False):
    steps_per_episode = rospy.get_param("ur3e_gym/steps_per_episode", 200)
    total_steps = 0
    successes = 0
    avg_steps = 0
    print_count = 0
    for i in range(num_tests):
        episode_return = 0.
        obs = env.reset()
        start_time = timeit.default_timer()
        collision = False
        for j in range(steps_per_episode):
            action = policy.get_action(obs, test=(not training))
            next_obs, reward, done, info = env.step(action)
            episode_return += reward
            obs = next_obs
            if info == "collision":
                collision = True
            if done:
                total_steps = (j+1)
                break
        #     if print_count >= 100:
        #         print("lastest dist", np.round(obs[:6], 4).tolist())
        #         print_count = 0
        #     print_count += 1
        # print("last dist", np.round(obs[:6], 4).tolist())
        if not collision and total_steps < steps_per_episode-1 and episode_return > -300:
            # if episode_return > 0 and total_steps < steps_per_episode-1:
            avg_steps += total_steps
            successes += 1
            if custom_path:
                env.straight_path(np.array([-0.40012212, -0.16659674,  0.37670753, -0.49865593,  0.49269791,  0.52005059, -0.48799429]), np.array([10., 0, 0, 0, 0, 0]), duration=2)  # elec1

        hz = 1. / ((timeit.default_timer() - start_time) / total_steps)
        # print("Total Epi: {0: 5} Episode Steps: {1: 5} Return: {2: 5.4f} Hz: {3: 5.2f}".format(
        #             i+1, total_steps, episode_return, hz))
        logger.info("Total Epi: {0: 5} Episode Steps: {1: 5} Return: {2: 5.4f} Hz: {3: 5.2f}".format(
                    i+1, total_steps, episode_return, hz))
    env.reset()
    # print("Successes:", successes,"of",num_tests)
    logger.info("Results: {0: 5} of {1:5}".format(successes, num_tests))
    logger.info("Avg steps: {0: 5}".format(avg_steps/float(successes+0.01)))
    # if successes > 0:
    # print ("Avg. steps:", avg_steps/float(successes) )


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('poldir', type=str, default=None, help='Policy directory')
    parser.add_argument('-c', '--common-test', action='store_true', help='Use common test parameters ')
    parser.add_argument('-n', type=int, help='number of tests', default=1)
    parser.add_argument('--custom-path', action='store_true', help='Execute custom path for successful tests')
    parser.add_argument('--training', action='store_true', help='Execute actions within random std')
    parser.add_argument('-r', '--real', action='store_true', help='Use real robot')
    parser.add_argument('-s', '--record', action='store_true', help='Store/Record the observations and actions of the agent')
    parser.add_argument('--tag', type=str, default='', help='Tag testing session')

    args = parser.parse_args()

    rospy.init_node('ur3e_tf2rl',
                    anonymous=True,
                    log_level=rospy.ERROR)

    clear_gym_params('ur3e_gym')
    clear_gym_params('ur3e_force_control')
    # rospy.set_param('ur3e_gym/test_mode', True)

    policy_dir = os.path.join(os. getcwd(), args.poldir)
    assert os.path.exists(policy_dir)

    env = load_env(policy_dir, args.common_test, args.real)

    actor_class = rospy.get_param("ur3e_gym/actor_class", "default")

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        actor_class=actor_class,
    )

    global logger
    logger = initialize_logger(log_tag="dummy2", save_log=False)
    if args.record:
        rospy.set_param('ur3e_gym/output_dir', policy_dir)
        logger = initialize_logger(filename=rospy.get_param("ur3e_gym/output_dir") + "/test-results.log", log_tag="test")
    logger.info('TAG ' + args.tag)
    checkpoint = tf.train.Checkpoint(policy=policy)
    _latest_path_ckpt = tf.train.latest_checkpoint(policy_dir)
    checkpoint.restore(_latest_path_ckpt).expect_partial()
    print("Restored {}".format(_latest_path_ckpt))

    test_policy(env, policy, args.n, args.custom_path, args.training)


if __name__ == "__main__":
    main()
