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

import datetime
import rospy
import numpy as np
from ur3e_openai.control.controller import Controller
from ur_control.compliant_controller import CompliantController

from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_control import transformations, spalg
import ur3e_openai.cost_utils as cost

from gym import spaces

from ur3e_openai.robot_envs.dual_ur3e_env import DualUR3eEnv
from ur3e_openai.control.parallel_controller import ParallelController
from ur3e_openai.control.admittance_controller import AdmittanceController
from ur3e_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose


class ArmState():
    def __init__(self, arm: CompliantController,
                 controller: Controller,
                 dt, reset_time, random_initial_pose, init_q,
                 rand_init_interval,
                 workspace,
                 randomize_desired_force_scale,
                 uncertainty_std,
                 randomize_desired_force,
                 fixed_uncertainty_error,
                 target_pose_uncertain,
                 true_target_pose,
                 target_pos
                 ):
        self.ur3e_arm = arm
        self.controller = controller
        self._previous_joints = None
        self.agent_control_dt = dt
        self.reset_time = reset_time
        self.random_initial_pose = random_initial_pose
        self.init_q = init_q
        self.rand_init_interval = rand_init_interval
        self.workspace = workspace
        self.randomize_desired_force_scale = randomize_desired_force_scale
        self.uncertainty_std = uncertainty_std
        self.randomize_desired_force = randomize_desired_force
        self.fixed_uncertainty_error = fixed_uncertainty_error
        self.target_pose_uncertain = target_pose_uncertain
        self.true_target_pose = true_target_pose
        self.target_pos = target_pos

    def get_points_and_vels(self, current_joint_angles):
        """
        Helper function that gets the cartesian positions
        and velocities from ROS."""

        if self._previous_joints is None:
            self._previous_joints = self.ur3e_arm.joint_angles()

        # Current position
        ee_pos_now = self.ur3e_arm.end_effector(joint_angles=current_joint_angles)

        # Last position
        ee_pos_last = self.ur3e_arm.end_effector(joint_angles=self._previous_joints)
        self._previous_joints = current_joint_angles  # update

        # Use the past position to get the present velocity.
        linear_velocity = (ee_pos_now[:3] - ee_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            ee_pos_now[3:], ee_pos_last[3:], self.agent_control_dt)
        velocity = np.concatenate((linear_velocity, angular_velocity))

        # Shift the present poistion by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        error = spalg.translation_rotation_error(self.target_pos, ee_pos_now)

        # scale error error, for more precise motion
        # (numerical error with small numbers?)
        error *= [1000, 1000, 1000, 1000., 1000., 1000.]

        return error, velocity

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        cpose = self.ur3e_arm.end_effector()
        deltax = np.array([0., 0., 0.02, 0., 0., 0.])
        cpose = transformations.pose_from_angular_velocity(cpose, deltax, dt=self.reset_time, rotated_frame=True)
        self.ur3e_arm.set_target_pose(pose=cpose,
                                      wait=True,
                                      t=self.reset_time)
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
        self._randomize_desired_force()
        self.max_distance = spalg.translation_rotation_error(self.ur3e_arm.end_effector(), self.target_pos) * 1000.
        self.max_dist = None

    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(self.ur3e_arm.end_effector(self.init_q), self.workspace, self.reset_time)
            self.rand_init_counter = 0
        self.rand_init_counter += 1

    def _randomize_desired_force(self, override=False):
        if self.randomize_desired_force:
            desired_force = np.zeros(6)
            desired_force[2] = np.abs(np.random.normal(scale=self.randomize_desired_force_scale))
            self.controller.target_force_torque = desired_force

    def _add_uncertainty_error(self):
        if self.target_pose_uncertain:
            if len(self.uncertainty_std) == 2:
                translation_error = np.random.normal(scale=self.uncertainty_std[0], size=3)
                translation_error[2] = 0.0
                rotation_error = np.random.normal(scale=self.uncertainty_std[1], size=3)
                rotation_error = np.deg2rad(rotation_error)
                error = np.concatenate([translation_error, rotation_error])
            elif len(self.uncertainty_std) == 6:
                if self.fixed_uncertainty_error:
                    error = self.uncertainty_std.copy()
                    error[3:] = np.deg2rad(error[3:])
                else:
                    translation_error = np.random.normal(scale=self.uncertainty_std[:3])
                    rotation_error = np.random.normal(scale=self.uncertainty_std[3:])
                    rotation_error = np.deg2rad(rotation_error)
                    error = np.concatenate([translation_error, rotation_error])
            else:
                print("Warning: invalid uncertanty error", self.uncertainty_std)
                return
            self.target_pos = transformations.transform_pose(self.true_target_pose, error, rotated_frame=True)


class DualUR3eTaskSpaceFTEnv(DualUR3eEnv):
    def __init__(self):

        self.cost_positive = False
        self.get_robot_params()

        DualUR3eEnv.__init__(self)

        self._init_controller()
        self._previous_joints = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None

        self.leftarm_state = ArmState(self.left_ur3e_arm,
                                      self.left_controller, 
                                      self.agent_control_dt,
                                      self.reset_time,
                                      self.random_initial_pose,
                                      self.l_init_q,
                                      self.rand_init_interval,
                                      self.workspace,
                                      self.randomize_desired_force_scale,
                                      self.uncertainty_std,
                                      self.randomize_desired_force,
                                      self.fixed_uncertainty_error,
                                      self.target_pose_uncertain,
                                      self.true_target_pose,
                                      self.left_target_pos)

        self.rightarm_state = ArmState(self.right_ur3e_arm,
                                      self.right_controller, 
                                      self.agent_control_dt,
                                      self.reset_time,
                                      self.random_initial_pose,
                                      self.l_init_q,
                                      self.rand_init_interval,
                                      self.workspace,
                                      self.randomize_desired_force_scale,
                                      self.uncertainty_std,
                                      self.randomize_desired_force,
                                      self.fixed_uncertainty_error,
                                      self.target_pose_uncertain,
                                      self.true_target_pose,
                                      self.right_target_pos)

        self.last_actions = np.zeros(self.n_actions)
        obs = self._get_obs()

        self.reward_threshold = 500.0

        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions, ),
                                       dtype='float32')

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs.shape,
                                            dtype='float32')

        self.trials = 1


        print("ACTION SPACES TYPE", (self.action_space))
        print("OBSERVATION SPACES TYPE", (self.observation_space))

    def get_robot_params(self):
        prefix = "ur3e_gym"
        load_param_vars(self, prefix)

        self.param_use_gazebo = False
        
        self.ft_topic = rospy.get_param(prefix + "/ft_topic", None)

        self.relative_to_ee = rospy.get_param(prefix + "/relative_to_ee", False)

        self.target_pose_uncertain = rospy.get_param(prefix + "/target_pose_uncertain", False)
        self.fixed_uncertainty_error = rospy.get_param(prefix + "/fixed_uncertainty_error", False)
        self.target_pose_uncertain_per_step = rospy.get_param(prefix + "/target_pose_uncertain_per_step", False)
        self.true_target_pose = rospy.get_param(prefix + "/target_pos", False)
        self.rand_seed = rospy.get_param(prefix + "/rand_seed", None)
        self.ft_hist = rospy.get_param(prefix + "/ft_hist", False)
        self.rand_init_interval = rospy.get_param(prefix + "/rand_init_interval", 5)
        self.rand_init_counter = 0
        self.rand_init_cpose = None
        self.insertion_direction = rospy.get_param(prefix + "/insertion_direction", 1)
        self.wrench_hist_size = rospy.get_param(prefix + "/wrench_hist_size", 12)
        self.randomize_desired_force = rospy.get_param(prefix + "/randomize_desired_force", False)
        self.randomize_desired_force_scale = rospy.get_param(prefix + "/randomize_desired_force_scale", 1)
        self.test_mode = rospy.get_param(prefix + "/test_mode", False)

    def _init_controller(self):
        """
            Initialize controller
            position: direct task-space control
            parallel position-force: force control with parallel approach
            admittance: impedance on all task-space directions
        """

        if self.controller_type == "parallel_position_force":
            self.left_controller = ParallelController(self.left_ur3e_arm, self.agent_control_dt)
            self.right_controller = ParallelController(self.right_ur3e_arm, self.agent_control_dt)
        elif self.controller_type == "admittance":
            self.left_controller = AdmittanceController(self.left_ur3e_arm, self.agent_control_dt)
            self.right_controller = AdmittanceController(self.right_ur3e_arm, self.agent_control_dt)
        else:
            raise Exception("Unsupported controller" + self.controller_type)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        l_joint_angles = self.left_ur3e_arm.joint_angles()
        l_ee_points, l_ee_velocities = self.leftarm_state.get_points_and_vels(l_joint_angles)
        r_joint_angles = self.right_ur3e_arm.joint_angles()
        r_ee_points, r_ee_velocities = self.rightarm_state.get_points_and_vels(r_joint_angles)

        obs = None

        l_desired_force_wrt_goal = -1.0 * spalg.convert_wrench(self.left_controller.target_force_torque, self.left_target_pos)
        r_desired_force_wrt_goal = -1.0 * spalg.convert_wrench(self.right_controller.target_force_torque, self.right_target_pos)
        if self.ft_hist:
            l_force_torque = (self.left_ur3e_arm.get_ee_wrench_hist(self.wrench_hist_size) - l_desired_force_wrt_goal) / self.left_controller.max_force_torque
            r_force_torque = (self.right_ur3e_arm.get_ee_wrench_hist(self.wrench_hist_size) - r_desired_force_wrt_goal) / self.right_controller.max_force_torque
            force_torque = np.concatenate([l_force_torque, r_force_torque])
            obs = np.concatenate([
                l_ee_points.ravel(),  # [6]
                l_ee_velocities.ravel(),  # [6]
                r_ee_points.ravel(),  # [6]
                r_ee_velocities.ravel(),  # [6]
                l_desired_force_wrt_goal[:3].ravel(),
                r_desired_force_wrt_goal[:3].ravel(),
                self.last_actions.ravel(),  # [14]
                force_torque.ravel(),  # [6]*24
            ])
        else:
            l_force_torque = (self.left_ur3e_arm.get_ee_wrench_hist(self.wrench_hist_size) - l_desired_force_wrt_goal) / self.left_controller.max_force_torque
            r_force_torque = (self.right_ur3e_arm.get_ee_wrench_hist(self.wrench_hist_size) - r_desired_force_wrt_goal) / self.right_controller.max_force_torque
            force_torque = np.concatenate([l_force_torque, r_force_torque])

            obs = np.concatenate([
                l_ee_points.ravel(),  # [6]
                l_ee_velocities.ravel(),  # [6]
                r_ee_points.ravel(),  # [6]
                r_ee_velocities.ravel(),  # [6]
                force_torque.ravel(),  # [6]
                self.last_actions.ravel(),  # [14]
            ])

        return obs.copy()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self._log()
        self.left_controller.reset()
        self.right_controller.reset()
        self.leftarm_state._set_init_pose()
        self.rightarm_state._set_init_pose()

    def _log(self):
        # Test
        # log_data = np.array([self.controller.force_control_model.update_data,self.controller.force_control_model.error_data])
        # print("Hellooo",log_data.shape)
        # logfile = rospy.get_param("ur3e_gym/output_dir") + "/log_" + \
        #             datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
        # np.save(logfile, log_data)
        if self.obs_logfile is None:
            try:
                self.obs_logfile = rospy.get_param("ur3e_gym/output_dir") + "/state_" + \
                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
            except Exception:
                return
        # save_log(self.obs_logfile, self.obs_per_step, self.reward_per_step, self.cost_ws)
        self.reward_per_step = []
        self.obs_per_step = []

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        state = []
        if self.ft_hist and not self.test_mode:
            ft_size = self.wrench_hist_size*6
            state = np.concatenate([
                observations[:-ft_size].ravel(),
                observations[-6:].ravel(),
                [self.action_result]
            ])
        else:
            state = np.concatenate([
                observations.ravel(),
                [self.action_result]
            ])
        self.obs_per_step.append([state])

        self.max_distance = self.leftarm_state.max_distance

        if self.reward_type == 'sparse':
            return cost.sparse(self, done)
        elif self.reward_type == 'zero':
            return 0
        elif self.reward_type == 'distance':
            return -1 * cost.distance(self, observations, done)
        elif self.reward_type == 'force':
            return cost.distance_force_action_step_goal(self, observations, done)
        else:
            raise AssertionError("Unknown reward function", self.reward_type)

    def _is_done(self, observations):
        if self.target_pose_uncertain_per_step:
            self._add_uncertainty_error()

        l_true_error = spalg.translation_rotation_error(self.left_target_pos, self.left_ur3e_arm.end_effector())
        l_true_error[:3] *= 1000.0
        l_true_error[3:] = np.rad2deg(l_true_error[3:])
        success = np.linalg.norm(l_true_error[:3], axis=-1) < self.distance_threshold
        self._log_message = "Final distance: " + str(np.round(l_true_error, 3)) + (' inserted!' if success else '')
        
        r_true_error = spalg.translation_rotation_error(self.right_target_pos, self.right_ur3e_arm.end_effector())
        r_true_error[:3] *= 1000.0
        r_true_error[3:] = np.rad2deg(r_true_error[3:])
        success = np.linalg.norm(r_true_error[:3], axis=-1) < self.distance_threshold
        self._log_message = "Final distance: " + str(np.round(r_true_error, 3)) + (' inserted!' if success else '')
        

        return success or self.action_result == FORCE_TORQUE_EXCEEDED

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)

    def _set_action(self, action):
        self.l_action_result = self.left_controller.act(action, self.left_target_pos)
        self.r_action_result = self.right_controller.act(action, self.right_target_pos)
