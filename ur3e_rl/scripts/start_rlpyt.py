#!/usr/bin/env python
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

"""

from rlpyt.samplers.serial.sampler import SerialSampler

from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from ur3e_openai.common import load_environment
from ur3e_openai.task_envs.task_commons import load_ros_params

from gym.envs.registration import register
import rospy
import sys


def build_and_train(env_id=None, run_ID=0, cuda_idx=None):

    # env = load_environment(env_id, timestep_limit_per_episode=100)

    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(100),
        eval_max_trajectories=50,
    )
    algo = SAC(batch_size=256,
               min_steps_learn=100,
               learning_rate=0.005,
               clip_grad_norm=1e9)  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2000,
        # seed=8465,
        log_interval_steps=100,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "UR3e_SAC"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse

    rospy.init_node('ur3e_learn_to_pick_cube_qlearn',
                    anonymous=True,
                    log_level=rospy.ERROR)
    load_ros_params(
        rospackage_name="ur3e_rl",
        rel_path_from_package_to_file="config",
        yaml_file_name="ur3e_ee_rlpyt.yaml")
    env_name = rospy.get_param('/ur3e/env_name')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default=env_name)
    parser.add_argument('--run_ID',
                        help='run identifier (logging)',
                        type=int,
                        default=0)
    parser.add_argument('--cuda_idx',
                        help='gpu to use ',
                        type=int,
                        default=None)
    print("my args", rospy.myargv()[1:])
    args = parser.parse_args(rospy.myargv()[1:])


    register(
        id=args.env_id,
        entry_point='ur3e_openai.task_envs.ur3e.peg_in_hole:UR3ePegInHoleEnv',
        max_episode_steps=100,
    )

    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
