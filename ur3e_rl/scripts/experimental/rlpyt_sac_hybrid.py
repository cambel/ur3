#!/usr/bin/env python
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

"""

import argparse
import rospy
import timeit
import pickle
import os.path as osp
import torch

import ur_gazebo.log as utils

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context, get_log_dir

from gym.envs.registration import register

color_log = utils.TextColors()


def build(env_id=None, run_id=0, cuda_idx=None, state_dict=None):

    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=100,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=0,
        eval_max_steps=int(100),
        eval_max_trajectories=50,
    )
    algo = SAC(
        batch_size=256,
        min_steps_learn=100,
        learning_rate=0.005,
        clip_grad_norm=99,
        discount=0.99,
        replay_size=int(1e6),
        replay_ratio=256,  # data_consumption / data_generation
        target_update_tau=0.005,  # tau=1 for hard update.
        target_update_interval=1,  # 1000 for hard update, 1 for soft.
        # OptimCls=torch.optim.Adam,
        # action_prior="gaussian",  # or "gaussian"
        reward_scale=1,
        target_entropy="auto",  # "auto", float, or None
        reparameterize=True,
        # policy_output_regularization=0.001,
        n_step_return=1,
        updates_per_sync=1,  # For async mode only.
        bootstrap_timelimit=True)

    # model = dict(hidden_sizes=[40, 40])
    # q_v_model = dict(hidden_sizes=[64, 64])
    agent = SacAgent(
        # model_kwargs=model,
        # q_model_kwargs=q_v_model,
        # v_model_kwargs=q_v_model,
        initial_model_state_dict=state_dict)

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=5000,
        seed=8393,
        log_interval_steps=100,
        affinity=dict(cuda_idx=cuda_idx),
    )

    return runner


def load_model(run_id):
    log_dir = "UR3e_SAC"
    folder = get_log_dir(log_dir, run_id)
    filename = osp.join(folder, "params.pkl")
    try:
        print("Loading model", filename)
        state_dict = torch.load(filename)["agent_state_dict"]
        color_log.ok("Model loaded")
        return state_dict
    except Exception:
        color_log.error("Model does not exists")
        return None


def train(env_id, run_id, runner):
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "UR3e_SAC"
    with logger_context(log_dir, run_id, name, config, snapshot_mode='last'):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default=None)
    parser.add_argument('--run_id',
                        help='run identifier (logging)',
                        type=int,
                        default=0)
    parser.add_argument('--cuda_idx',
                        help='gpu to use ',
                        type=int,
                        default=None)
    parser.add_argument('--exec',
                        help='exec policy',
                        action='store_true')
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('ur3e_hybrid',
                    anonymous=True,
                    log_level=rospy.ERROR)

    env_id = args.env_id if args.env_id is not None else 'UR3eHybridControlEnv-v0'
    run_id = args.run_id
    cuda_idx = args.cuda_idx

    start_time = timeit.default_timer()
    register(
        id=env_id,
        entry_point='ur3e_openai.task_envs.hybrid_control:UR3eHybridControlEnv',
        max_episode_steps=100,
    )

    # state_dict = load_model(run_id)
    state_dict = None
    if args.exec:
        test(env_id=env_id, run_id=run_id, cuda_idx=cuda_idx, state_dict=state_dict)
    else:
        runner = build(
            env_id=env_id,
            run_id=run_id,
            cuda_idx=cuda_idx,
            state_dict=state_dict,
        )
        train(
            env_id=env_id,
            run_id=run_id,
            runner=runner,
        )

        end_time = timeit.default_timer() - start_time
        print("Params Training time: ", end_time / 60., "min")
