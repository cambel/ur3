#!/usr/bin/env python
# The MIT License (MIT)
#
# Copyright (c) 2018-2022 Cristian C Beltran-Hernandez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian C Beltran-Hernandez

from gym.envs.registration import register
from gym import envs


def register_environment(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # Task-Robot Envs

    result = True

    if task_env == 'UR3eJointSpaceEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_joint_space:UR3eJointSpaceEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eTaskSpaceEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_task_space:UR3eTaskSpaceEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eTaskSpaceFTEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_task_space_ft:UR3eTaskSpaceFTEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePegInHoleEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_peg_in_hole:UR3ePegInHoleEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eParallelMidpointsEnv-v1':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_midpoints:UR3eMidpointsEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePipeCoatingEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.dual_ur3e_pipe_coating:UR3ePipeCoatingEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eDualTaskPipeCoatingEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.dual_task_pipe_coating:UR3eDualTaskPipeCoatingEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'DualUR3eTaskSpaceFTEnv-v0':

        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.dual_ur3e_task_space_ft:DualUR3eTaskSpaceFTEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eForceControlEnv-v0':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.ur3e_force_control:UR3eForceControlEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePushButtonEnv-v0':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.force_control.push_button:UR3ePushButtonEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePushBlockEnv-v0':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.force_control.push_block:UR3ePushBlockEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePegInHoleEnv-v1':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.force_control.peg_in_hole:UR3ePegInHoleEnv2',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3eSlicingEnv-v1':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.force_control.slicing:UR3eSlicingEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'UR3ePokingEnv-v1':
        register(
            id=task_env,
            entry_point='ur3e_openai.task_envs.force_control.poking:UR3ePokingEnv',
            max_episode_steps=max_episode_steps,
        )


    # Add here your Task Envs to be registered
    else:
        result = False
    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = get_all_registered_envs()
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result


def get_all_registered_envs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids
