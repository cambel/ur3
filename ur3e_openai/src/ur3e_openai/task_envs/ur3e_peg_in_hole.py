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

import rospy
import numpy as np


from ur3e_openai.task_envs.ur3e_task_space_ft import UR3eTaskSpaceFTEnv
from ur3e_openai.robot_envs.utils import load_param_vars, save_log, peg_in_hole_models, randomize_initial_pose,\
    get_value_from_range, get_conical_helix_trajectory, create_gazebo_marker
from ur_control import transformations, spalg, conversions
from pyquaternion import Quaternion
from ur_gazebo.gazebo_spawner import GazeboModels


class UR3ePegInHoleEnv(UR3eTaskSpaceFTEnv):
    """ Peg in Hole environment """

    def __init__(self):
        UR3eTaskSpaceFTEnv.__init__(self)
        self.__load_env_params()
        if self.gazebo_models:
            self.gazebo_spawner = GazeboModels('ur3_gazebo')

    def __load_env_params(self):
        prefix = "ur3e_gym"
        # Gazebo spawner parameters
        self.gazebo_models = rospy.get_param(prefix + "/gazebo_models", False)
        self.gazebo_model_stiffness = rospy.get_param(prefix + "/gazebo_model_stiffness", 1e5)
        self.gazebo_models_episode_num = rospy.get_param(prefix + "/gazebo_models_episode_num", 100)
        self.gazebo_markers = rospy.get_param(prefix + "/gazebo_markers", False)
        self.gazebo_stiff_upper_limit = rospy.get_param(prefix + "/gazebo_stiff_upper_limit", 5e5)
        self.gazebo_stiff_lower_limit = rospy.get_param(prefix + "/gazebo_stiff_lower_limit", 5e4)
        self.gazebo_params = {'stiffness': self.gazebo_model_stiffness,
                              'stiff_upper_limit': self.gazebo_stiff_upper_limit,
                              'stiff_lower_limit': self.gazebo_stiff_lower_limit}
        self.gazebo_spawner_counter = -1

        # Initial conditions
        self.changable_goal = rospy.get_param(prefix + "/changable_goal", False)
        self.goal_changer_interval = rospy.get_param(prefix + "/goal_changer_interval", 100)
        self.goal_changer_counter = -1

        self.traj_max_steps = rospy.get_param(prefix + "/traj_max_steps", 300)
        self.traj_revolutions = rospy.get_param(prefix + "/traj_revolutions", 3.0)
        self.use_conical_helix_trajectory = rospy.get_param(prefix + "/use_conical_helix_trajectory", False)
        self.use_conical_param = rospy.get_param(prefix + "/use_conical_param", False)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self._log()
        cpose = self.ur3e_arm.end_effector()
        deltax = np.array([0., 0., 0.02, 0., 0., 0.])
        cpose = transformations.pose_from_angular_velocity(cpose, deltax, dt=self.reset_time, rotated_frame=True)
        self.ur3e_arm.set_target_pose(pose=cpose,
                                      wait=True,
                                      t=self.reset_time)
        self._change_goal()
        if self.changable_goal and self.goal_changer_counter >= 0:
            self._add_uncertainty_error()
        else:
            self._add_uncertainty_error()

        if self.random_initial_pose:
            self._randomize_initial_pose()
            self.ur3e_arm.set_target_pose(pose=self.rand_init_cpose,
                                          wait=True,
                                          t=self.reset_time)
        else:
            qc = self.init_q
            self.ur3e_arm.set_joint_positions(position=qc,
                                              wait=True,
                                              t=self.reset_time)
        self.ur3e_arm.set_wrench_offset(True)
        self.controller.reset()
        self.max_distance = spalg.translation_rotation_error(self.ur3e_arm.end_effector(), self.target_pos) * 1000.
        self.max_dist = None
        self._gazebo_spawner()
        if self.use_conical_helix_trajectory:
            self._conical_helix_trajectory(self.traj_max_steps, self.traj_revolutions)

    def _change_goal(self):
        if self.changable_goal and (self.goal_changer_counter >= self.goal_changer_interval or self.goal_changer_counter == -1):
            index = np.random.randint(0, len(self.goals), size=1)[0]
            self.init_q = self.goals[index]
            self.true_target_pose = np.array(conversions.transform_end_effector(
                self.ur3e_arm.end_effector(joint_angles=self.init_q), [0., 0., 0.02, 0, 0, 0, 1]))
            self.target_pos = self.true_target_pose.copy()
            self._add_uncertainty_error()
            self.ur3e_arm.set_joint_positions(position=self.init_q, wait=True, t=self.reset_time)
            self.goal_changer_counter = 0
            self._gazebo_spawner(True)
            self._randomize_initial_pose(True)
        self.goal_changer_counter += 1

    def _gazebo_spawner(self, override=False):
        if self.gazebo_models and (self.gazebo_spawner_counter > self.gazebo_models_episode_num
                                   or self.gazebo_spawner_counter == -1 or override):
            print("updating gazebo models")
            board_pose = np.array(conversions.transform_end_effector(self.true_target_pose, [0., 0., 0.03, 0, 1, 0, 0]))
            markers = []
            if self.gazebo_markers:
                markers.append(conversions.transform_end_effector(self.true_target_pose, [0., 0., -0.01, 0, 1, 0, 0]))
            models = peg_in_hole_models(board_pose, self.gazebo_params, marker_poses=markers)
            self.gazebo_spawner.reset_models(models)
            self.gazebo_spawner_counter = 0
        self.gazebo_spawner_counter += 1
        self._gazebo_update_marker()

    def _gazebo_update_marker(self):
        if self.gazebo_models and self.gazebo_markers:
            marker_pose = conversions.transform_end_effector(self.target_pos, [0., 0., -0.01, 0, 1, 0, 0])
            marker = create_gazebo_marker(marker_pose, "base_link", marker_id="marker%s" % 2)
            self.gazebo_spawner.reset_model(marker)

    def _conical_helix_trajectory(self, steps, revolutions):
        # initial_pose = self.ur3e_arm.end_effector()[:3]
        initial_pose = self.rand_init_cpose[:3]
        final_pose = self.target_pos[:3]

        target_q = transformations.vector_to_pyquaternion(self.target_pos[3:])

        p1 = target_q.rotate(initial_pose - final_pose)
        p2 = np.zeros(3)

        traj = get_conical_helix_trajectory(p1, p2, steps, revolutions)
        traj = np.apply_along_axis(target_q.rotate, 1, traj)
        self.base_trajectory = traj + final_pose

    def _set_action(self, action):
        self.last_actions = action.copy()
        if self.use_conical_param:
            revolutions = get_value_from_range(action[-1], 3, 3)
            self._conical_helix_trajectory(self.traj_max_steps, revolutions)

        if self.use_conical_helix_trajectory:
            if self.step_count < len(self.base_trajectory):
                target = self.base_trajectory[self.step_count]
            else:
                target = self.base_trajectory[-1]
            target = np.concatenate([target, self.target_pos[3:]])
        else:
            target = self.target_pos

        if self.use_conical_param:
            self.action_result = self.controller.act(action[:-1], target)
        else:
            self.action_result = self.controller.act(action, target)
