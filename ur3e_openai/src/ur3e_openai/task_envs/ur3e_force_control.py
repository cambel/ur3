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

from ur3e_openai.initialize_logger import initialize_logger
from ur_gazebo.model import Model
from ur_gazebo.basic_models import SPHERE
from ur_gazebo.gazebo_spawner import GazeboModels
from ur3e_openai.robot_envs.utils import load_param_vars, save_log, randomize_pose, simple_random
from ur3e_openai.control.admittance_controller import AdmittanceController
from ur3e_openai.control.compliance_controller import ComplianceController
from ur3e_openai.control.parallel_controller import ParallelController
from ur3e_openai.robot_envs import ur3e_env
from gym import spaces
import ur3e_openai.cost_utils as cost
from ur_control import conversions, transformations, spalg
from ur_control.constants import FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND
import numpy as np
import rospy
import datetime
import tf


class UR3eForceControlEnv(ur3e_env.UR3eEnv):
    """
        UR3e environment for learning force control

        Inputs: position/velocity command, target force
        Observations: properception (eef position, eef velocity, force/torque, time-from-start)
        Actions: controller parameters
        Modes:
        - Specific Target force
        - Force constraints (limit force to a certain range, non specific values)

    """

    def __init__(self):

        self.get_robot_params()

        ur3e_env.UR3eEnv.__init__(self)

        self.tf_listener = tf.TransformListener()
        rospy.sleep(1)

        self._init_controller()
        self.previous_ee_pose = None
        self.previous_pose = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.reward_details_per_step = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None
        self.difficulty_ratio = 0.1

        self.last_actions = np.zeros(self.n_actions)
        self.object_current_pose = None
        self.out_of_workspace = False

        obs = self._get_obs(verbose=True)

        self.reward_threshold = 500.0

        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions, ),
                                       dtype='float32')

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs.shape,
                                            dtype='float32')

        self.logger = initialize_logger(log_tag="dummy", save_log=False)

        if not self.real_robot:
            self.spawner = GazeboModels('ur3_gazebo')
            self.target_model = Model("target", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
                "ball", 0.01, "GreenTransparent"), reference_frame="base_link")
            self.start_model = Model("start", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
                "ball", 0.01, "RedTransparent"), reference_frame="base_link")

        print("ACTION SPACES TYPE", (self.action_space.shape))
        print("OBSERVATION SPACES TYPE", (self.observation_space.shape))

    def get_robot_params(self):
        prefix = "ur3e_gym"
        load_param_vars(self, prefix)

        self.use_gazebo_sim = rospy.get_param(prefix + "/use_gazebo_sim", False)
        self.test_mode = rospy.get_param(prefix + "/test_mode", False)
        if not self.test_mode:
            self.curriculum_learning = rospy.get_param(prefix + "/curriculum_learning", False)
            self.reward_based_on_cl = rospy.get_param(prefix + "/reward_based_on_cl", 0)
        else:
            self.curriculum_learning = False
            self.reward_based_on_cl = False

        self.real_robot = rospy.get_param(prefix + "/real_robot", False)
        self.reset_gazebo_robot = rospy.get_param(prefix + "/reset_robot", False)
        self.action_type = rospy.get_param(prefix + "/action_type", None)

        # Observations
        self.ft_hist = rospy.get_param(prefix + "/ft_hist", False)
        self.target_force_as_obs = rospy.get_param(prefix + "/target_force_as_obs", False)
        self.duration_as_obs = rospy.get_param(prefix + "/duration_as_obs", False)
        self.target_duration = rospy.get_param(prefix + "/target_duration", 0)
        self.last_action_as_obs = rospy.get_param(prefix + "/last_action_as_obs", False)
        self.jerkiness_as_obs = rospy.get_param(prefix + "/jerkiness_as_obs", False)
        self.completion_as_obs = rospy.get_param(prefix + "/completion_as_obs", False)

        self.target_dims = rospy.get_param(prefix + "/target_dims", [0, 1, 2, 3, 4, 5])
        
        max_distance = rospy.get_param(prefix + "/max_distance", [0.05, 0.05, 0.05, 0.785398, 0.785398, 0.785398])
        max_velocity = rospy.get_param(prefix + "/max_velocity", [0.5, 0.5, 0.5, 0.785398, 0.785398, 0.785398])

        self.max_velocity = np.array([max_velocity[i] for i in self.target_dims])
        self.max_distance = np.array([max_distance[i] for i in self.target_dims])

        self.random_initial_pose = rospy.get_param(prefix + "/random_initial_pose", False)

        self.relative_to_ee = rospy.get_param(prefix + "/relative_to_ee", False)

        self.current_target_pose_uncertain = rospy.get_param(prefix + "/target_pose_uncertain", False)
        self.fixed_uncertainty_error = rospy.get_param(prefix + "/fixed_uncertainty_error", False)
        self.current_target_pose_uncertain_per_step = rospy.get_param(prefix + "/target_pose_uncertain_per_step", False)
        self.current_target_pose = rospy.get_param(prefix + "/target_pose", False)

        self.rand_interval = rospy.get_param(prefix + "/rand_interval", 1)

        self.update_target = rospy.get_param(prefix + "/update_target", True)

        self.rand_initial_pose = None
        self.simple_rand_initial_pose = False
        self.rand_target_counter = 0
        self.rand_target_pose = None
        self.insertion_direction = rospy.get_param(prefix + "/insertion_direction", 1)
        if self.ft_hist:
            self.wrench_hist_size = rospy.get_param(prefix + "/wrench_hist_size", 12)
        else:
            self.wrench_hist_size = 1
        self.randomize_desired_force = rospy.get_param(prefix + "/randomize_desired_force", False)

        self.object_centric = rospy.get_param(prefix + "/object_centric", False)
        self.ee_centric = rospy.get_param(prefix + "/ee_centric", True)
        self.object_name = rospy.get_param(prefix + "/object_name", "target_board_tmp")

        self.cumulative_episode_num = rospy.get_param(prefix + "/cumulative_episode_num", 0)
        self.curriculum_level = rospy.get_param(prefix + "/initial_curriculum_level", 0.1)
        self.curriculum_level_step = rospy.get_param(prefix + "/curriculum_level_step", 0.1)
        self.progressive_cl = rospy.get_param(prefix + "/progressive_cl", False)
        self.two_steps = rospy.get_param(prefix + "/two_steps", False)
        self.normalize_velocity = rospy.get_param(prefix + "/normalize_velocity", False)

        self.termination_on_negative_reward = rospy.get_param(prefix + "/termination_on_negative_reward", False)
        self.termination_reward_threshold = rospy.get_param(prefix + "/termination_reward_threshold", -100)
        
        self.use_dynamic_rewards = rospy.get_param(prefix + "/use_dynamic_rewards", False)

        self.max_steps = rospy.get_param(prefix + "/max_steps", 50000)     

    def _init_controller(self):
        """
            Initialize controller
            position: direct task-space control
            parallel position-force: force control with parallel approach
            admittance: impedance on all task-space directions
        """

        if self.controller_type == "parallel_position_force":
            self.controller = ParallelController(
                self.ur3e_arm, self.agent_control_dt, self.robot_control_dt, self.n_actions, self.object_centric)
            self.controller.ee_centric = self.ee_centric
        elif self.controller_type == "admittance":
            self.controller = AdmittanceController(
                self.ur3e_arm, self.agent_control_dt)
        elif self.controller_type == "cartesian_compliance":
            self.controller = ComplianceController(
                self.ur3e_arm, self.agent_control_dt, self.robot_control_dt, self.n_actions
            )
        else:
            raise Exception("Unsupported controller" + self.controller_type)

    def _get_obs(self, verbose=False):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have access
        :return: observations
        """
        ee_points, ee_velocities, ee_acceleration, ee_jerkiness = self.get_end_effector_relative_position_and_velocity()

        # Normalize distance error
        ee_points /= self.max_distance

        if self.normalize_velocity:
            ee_velocities /= self.max_velocity

        ee_jerkiness /= 500.0  # normalize, no clipping

        target_force = np.zeros(6)

        last_action = np.array([])
        if self.last_action_as_obs:
            last_action = self.last_actions

        jerkiness_obs = np.array([])
        if self.jerkiness_as_obs:
            jerkiness_obs = ee_jerkiness

        if self.ft_hist:
            force_torque = (self.ur3e_arm.get_ee_wrench_hist(
                self.wrench_hist_size) - target_force) / self.controller.max_force_torque
            force_torque = np.array([force_torque[:, i] for i in self.target_dims])
        else:
            force_torque = (self.ur3e_arm.get_ee_wrench() -
                            target_force) / self.controller.max_force_torque
            force_torque = np.array([force_torque[i] for i in self.target_dims])

        reward_weights = np.array([])
        if self.use_dynamic_rewards:
            reward_weights = self.update_dynamic_reward_weights()

        obs = np.concatenate([
            ee_points.ravel(),  # [6]
            ee_velocities.ravel(),  # [6]
            jerkiness_obs.ravel(),  # [6]
            last_action.ravel(),  # None or [*]
            reward_weights.ravel(),
            force_torque.ravel(),  # [6] or [6]*24
        ])

        if verbose:
            rospy.loginfo("===== Observations =====")
            rospy.loginfo("ee_points %s \n \
                        ee_velocities %s \n \
                        jerkiness_obs %s \n \
                        last_action %s \n \
                        reward_weights %s \n \
                        force_torque %s" % (ee_points.shape, ee_velocities.shape, jerkiness_obs.shape, 
                                            last_action.shape, reward_weights.shape, force_torque.shape))

        return obs.copy()

    def update_dynamic_reward_weights(self):
        """
        update the weights of the rewards following consecutive gaussian curves
        """
        x = self.total_steps / self.max_steps
        sigma1, offset_start1, offset_end1 = 0.25, 0.1, 0.3
        sigma2, offset_start2, offset_end2 = 0.25, 0.1, 0.3
        sigma3, offset_start3, offset_end3 = 0.25, 0.1, 0.3
        
        # self.w_dist = self.mu1
        # self.w_force = self.mu2
        # self.w_jerk = self.mu3
        self.w_dist = np.exp((-(x-self.mu1)**2)/(2*sigma1**2))
        self.w_force = np.exp((-(x-self.mu2)**2)/(2*sigma2**2))
        self.w_jerk = np.exp((-(x-self.mu3)**2)/(2*sigma3**2))
        if x > self.mu1 : self.w_dist = max(offset_end1, self.w_dist) 
        elif x < self.mu1 : self.w_dist = max(offset_start1, self.w_dist) 
        if x > self.mu2 : self.w_force = max(offset_end2, self.w_force) 
        elif x < self.mu2 : self.w_force = max(offset_start2, self.w_force) 
        if x > self.mu3 : self.w_w_jerk = max(offset_end3, self.w_jerk)
        elif x < self.mu3 : self.w_jerk = max(offset_start3, self.w_jerk)  

        # print([self.w_dist, self.w_force, self.w_jerk])
        return np.array([self.w_dist, self.w_force, self.w_jerk])

    def get_end_effector_relative_position_and_velocity(self):
        """
        and velocities of the end effector with respect to the target pose
        """

        if self.previous_ee_pose is None:
            self.previous_ee_pose = self.ur3e_arm.end_effector()
            self.previous_velocity = np.zeros(6)
            self.previous_acceleration = np.zeros(6)

        # Current position
        current_ee_pose = self.ur3e_arm.end_effector()

        # Use the past position to get the present velocity.
        linear_velocity = (current_ee_pose[:3] - self.previous_ee_pose[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(current_ee_pose[3:], self.previous_ee_pose[3:], self.agent_control_dt)
        current_velocity = np.concatenate((linear_velocity, angular_velocity))

        # Get velocity and acceleration using finite difference approximation
        current_acceleration = (current_velocity - self.previous_velocity) / self.agent_control_dt

        # Estimate jerkiness using finite difference approximation
        current_jerkiness = (current_acceleration - self.previous_acceleration) / self.agent_control_dt

        # Update variables
        self.previous_ee_pose = current_ee_pose
        self.previous_velocity = np.copy(current_velocity)
        self.previous_acceleration = np.copy(current_acceleration)

        # Shift the present position by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        error = spalg.translation_rotation_error(self.current_target_pose, current_ee_pose)

        # Extract only directions of interest
        if self.target_dims is not None:
            error = np.array([error[i] for i in self.target_dims])
            current_velocity = np.array([current_velocity[i] for i in self.target_dims])
            current_acceleration = np.array([current_acceleration[i] for i in self.target_dims])
            current_jerkiness = np.array([current_jerkiness[i] for i in self.target_dims])

        return error, current_velocity, current_acceleration, current_jerkiness

    def _set_init_pose(self):
        """
        Define initial pose if random pose is desired.
        Otherwise, use manually defined initial pose.
        Then move the Robot to its initial pose
        """
        if self.random_initial_pose:
            qc = self.init_q
            self.ur3e_arm.set_joint_positions(position=qc,
                                              wait=True,
                                              t=self.reset_time)
            # print("center q", np.round(self.ur3e_arm.end_effector(), 4).tolist())
            res = IK_NOT_FOUND
            while (res == IK_NOT_FOUND):
                self._randomize_initial_pose()
                res = self.ur3e_arm.set_target_pose(pose=self.rand_initial_pose,
                                                    wait=True,
                                                    t=self.reset_time)
        else:
            # Manually define a joint initial configuration
            qc = self.init_q
            self.ur3e_arm.set_joint_positions(position=qc,
                                              wait=True,
                                              t=self.reset_time)
        self.update_scene()
        self.update_target_pose()
        self.ur3e_arm.zero_ft_sensor()
        self.controller.reset()
        self.controller.start()

    def update_target_pose(self):
        self.current_target_pose = rospy.get_param("ur3e_gym/target_pose", False)

        if not self.real_robot and self.update_target:
            if self.object_centric:
                if self.ee_centric:
                    translation, rotation = self.tf_listener.lookupTransform(
                        self.ur3e_arm.base_link, self.object_name, rospy.Time(0))
                    self.object_centric_transform = self.tf_listener.fromTranslationRotation(translation, rotation)
                else:
                    translation, rotation = self.tf_listener.lookupTransform(
                        self.object_name, self.ur3e_arm.base_link, rospy.Time(0))
                    self.object_centric_transform = self.tf_listener.fromTranslationRotation(translation, rotation)
                self.controller.ur3e_arm.object_centric_transform = self.object_centric_transform
                self.object_current_pose = np.array([0, 0, 0, 0, 0, 0, 1])
            else:
                obj_pose_msg = conversions.to_pose_stamped(self.object_name, [0, 0, 0, 0, 0, 0, 1])
                obj_base_link_pose_msg = self.tf_listener.transformPose(self.ur3e_arm.base_link, obj_pose_msg)
                self.object_current_pose = conversions.from_pose_to_list(obj_base_link_pose_msg.pose)
            self.current_target_pose = self.object_current_pose

        if self.uncertainty_error:
            rand = self.rng.random(size=6)
            scale = 1.0 if not self.curriculum_learning else self.curriculum_level
            uncertainty_range = [[-1*p*scale, p*scale] for p in self.uncertainty_error_max_range]
            rand = np.array([np.interp(rand[i], [0, 1.], uncertainty_range[i]) for i in range(6)])
            self.current_target_pose[:3] += rand[:3]
            temp = transformations.euler_from_quaternion(self.current_target_pose[3:]) + rand[3:]
            self.current_target_pose[3:] = transformations.quaternion_from_euler(*temp)
            # print("uncertainty", (np.round(rand, 4)).tolist())

    def update_scene(self):
        self.start_model.set_pose(self.ur3e_arm.end_effector())
        self.spawner.update_model_state(self.start_model)

        if self.random_target_pose:
            self.current_target_pose = self._randomize_target_pose()
        self.target_model.set_pose(self.current_target_pose)
        self.spawner.update_model_state(self.target_model)

    def _randomize_initial_pose(self):
        if self.simple_rand_initial_pose:
            rand_step = simple_random(self.workspace, self.np_random)
            self.rand_initial_pose = transformations.transform_pose(self.ur3e_arm.end_effector(self.init_q), rand_step)
        else:
            self.rand_initial_pose = randomize_pose(self.ur3e_arm.end_effector(self.init_q),
                                                    self.workspace, self.reset_time, rng=self.np_random)

    def _randomize_target_pose(self):
        if self.rand_target_pose is None or self.rand_target_counter >= self.rand_interval:
            self.rand_target_pose = randomize_pose(
                self.target_pose, self.workspace, self.reset_time, self.np_random)
            self.rand_target_counter = 0
        self.rand_target_counter += 1
        return self.rand_target_pose

    def _log(self):
        if self.obs_logfile is None:
            try:
                self.obs_logfile = rospy.get_param("ur3e_gym/output_dir") + "/state_" + \
                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
                if self.test_mode:
                    self.logger = initialize_logger(filename=rospy.get_param("ur3e_gym/output_dir") + "/test_console.log", log_tag="ur3e_openai_test")
                else:
                    self.logger = initialize_logger(filename=rospy.get_param("ur3e_gym/output_dir") + "/console.log")
            except Exception:
                return

        if len(self.obs_per_step) == 0:
            return

        # data = []
        # # if self.test_mode:
        # #     data.append(self.obs_per_step)
        # # else:
        # #     data.append([])
        # data.append([])
        # data.append(self.reward_per_step)
        # data.append(self.reward_details_per_step)

        # try:
        #     tmp = np.load(self.obs_logfile, allow_pickle=True).tolist()
        #     tmp.append(data)
        #     np.save(self.obs_logfile, tmp)
        #     tmp = None
        # except IOError:
        #     np.save(self.obs_logfile, [data], allow_pickle=True)

        # self.reward_per_step = []
        # self.obs_per_step = []
        # self.reward_details_per_step = []

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        state = []
        ft_size = self.wrench_hist_size*6
        state = np.concatenate([
            observations[:-ft_size].ravel(),
            observations[-6:].ravel(),
            [self.action_result]
        ])
        # if self.ft_hist and not self.test_mode:
        #     ft_size = self.wrench_hist_size*6
        #     state = np.concatenate([
        #         observations[:-ft_size].ravel(),
        #         observations[-6:].ravel(),
        #         [self.action_result]
        #     ])
        # else:
        #     state = np.concatenate([
        #         observations.ravel(),
        #         [self.action_result]
        #     ])
        # self.obs_per_step.append([state])
        self.obs_per_step = []

        reward = 0
        reward_details = []
        if self.reward_type == 'sparse':
            r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
            r_sparse = 1.0 if done else 0.0
            reward = r_sparse + r_collision
            reward_details = [r_sparse, r_collision]
        elif self.reward_type == 'slicing':  # position - target force
            reward, reward_details = cost.slicing(self, observations, done)
        elif self.reward_type == 'slicing_with_vel':  # position - target force
            reward, reward_details = cost.slicing_with_vel(self, observations, done)
        elif self.reward_type == 'peg-in-hole':  # position - target force
            reward, reward_details = cost.peg_in_hole(self, observations, done)
        elif self.reward_type == 'dense-pft':  # position - target force
            reward, reward_details = cost.dense_pft(self, observations, done)
        # elif self.reward_type == 'dense-pvfr': # position & velocity - target force
        elif self.reward_type == 'dense-pdft':  # position & duration - target force
            reward, reward_details = cost.dense_pdft(self, observations, done)
        # elif self.reward_type == 'dense-pvfr': # position & velocity - target force
        elif self.reward_type == 'dense-distance':
            reward, reward_details = cost.dense_distance(self, observations, done)
        elif self.reward_type == 'dense-distance-force':
            reward, reward_details = cost.dense_distance_force(self, observations, done)
        elif self.reward_type == 'dense-distance-velocity-force':
            reward, reward_details = cost.dense_distance_velocity_force(self, observations, done)
        elif self.reward_type == 'dense-factors':
            reward, reward_details = cost.dense_factors(self, observations, done)
        else:
            raise AssertionError("Unknown reward function", self.reward_type)

        if self.reward_based_on_cl:
            reward *= self.difficulty_ratio

        reward += 0.0 if not self.out_of_workspace else self.cost_collision
        
        # self.reward_per_step.append(reward)
        # self.reward_details_per_step.append(reward_details)
        
        return reward, reward_details

    def _is_done(self, observations):
        pose_error = observations[:6]
        position_reached = np.all(pose_error[:3] < self.position_threshold)
        orientation_reached = np.all(pose_error[3:] < self.orientation_threshold)
        if position_reached and orientation_reached:
            rospy.loginfo("Goal Reached! dist: %s" % (pose_error))

        if self.termination_on_negative_reward and self.cumulated_episode_reward <= self.termination_reward_threshold:
            rospy.loginfo("Fail: %s" % (pose_error))
            self.controller.stop()
            return True

        done = (position_reached and orientation_reached) \
            or self.action_result == FORCE_TORQUE_EXCEEDED
            
        if done:
            self.controller.stop()
        return done  # Stop on collision

    def _init_env_variables(self):
        self._log()
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)

    def _set_action(self, action):
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose)
