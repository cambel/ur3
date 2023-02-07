
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.robot_envs.utils import get_board_color
from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_gazebo.basic_models import get_box_model, get_cucumber_model
from ur_gazebo.model import Model


def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


class UR3eSlicingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        if not self.real_robot:
            # string_model = get_box_model("cube", self.object_size, color=[0, 1, 0, 0],
            #                              mass=self.object_mass, mu=self.object_friction, kp=self.object_stiffness)
            string_model = get_cucumber_model(kp=self.object_stiffness, soft_cfm=self.object_soft_cfm, soft_erp=self.object_soft_erp)
            self.box_model = Model("block", self.object_initial_pose, file_type="string",
                                   string_model=string_model, model_id="target_block", reference_frame="o2ac_ground")

    def __load_env_params(self):
        prefix = "ur3e_gym"

        # Gazebo spawner parameters
        self.randomize_object_properties = rospy.get_param(prefix + "/randomize_object_properties", False)
        self.object_initial_pose = rospy.get_param(prefix + "/object_initial_pose", [])
        self.object_stiffness = rospy.get_param(prefix + "/object_stiffness", 1e5)
        self.object_mu = rospy.get_param(prefix + "/object_mu", 1.0)
        self.object_mu2 = rospy.get_param(prefix + "/object_mu2", 1.0)
        self.object_soft_erp = rospy.get_param(prefix + "/object_soft_erp", 0.01)
        self.object_soft_cfm = rospy.get_param(prefix + "/object_soft_cfm", 0.2)
        self.object_size = rospy.get_param(prefix + "/object_size", 0.03)
        self.object_mass = rospy.get_param(prefix + "/object_mass", 1.0)
        self.object_friction = rospy.get_param(prefix + "/object_friction", 1)

        self.uncertainty_error = rospy.get_param(prefix + "/uncertainty_error", False)
        self.uncertainty_error_max_range = rospy.get_param(prefix + "/uncertainty_error_max_range", [0, 0, 0, 0, 0, 0])

        self.curriculum_level = rospy.get_param(prefix + "/initial_curriculum_level", 0.1)
        self.max_mu_range = rospy.get_param(prefix + "/max_mu_range", [1, 4])
        self.max_stiffness_range = rospy.get_param(prefix + "/max_stiffness_range", [5e5, 1e6])
        self.max_scale_range = rospy.get_param(prefix + "/max_scale_range", [1.05, 0.98])

        self.update_target = rospy.get_param(prefix + "/update_target", False)

        # How often to generate a new model, number of episodes
        self.refresh_rate = rospy.get_param(prefix + "/refresh_rate", False)
        self.normal_randomization = rospy.get_param(prefix + "/normal_randomization", True)
        self.basic_randomization = rospy.get_param(prefix + "/basic_randomization", False)
        self.random_type = rospy.get_param(prefix + "/random_type", "uniform")
        self.cl_upgrade_level = rospy.get_param(prefix + "/cl_upgrade_level", 0.8)
        self.cl_downgrade_level = rospy.get_param(prefix + "/cl_downgrade_level", 0.2)
        print(">>>>> ", self.random_type, self.curriculum_learning,
              self.progressive_cl, self.reward_based_on_cl, " <<<<<<")

    def _set_init_pose(self):
        if not self.real_robot:
            self.set_environment_conditions()

        reset_time = 1.0 if not self.real_robot else 5.0
        self.ur3e_arm.move_relative([0, 0, 0.03, 0, 0, 0], duration=reset_time, relative_to_tcp=False)

        UR3eForceControlEnv._set_init_pose(self)

    def update_scene(self):
        self.stage = 0
        block_pose = self.object_initial_pose  # self.randomize_block_position()
        if self.randomize_object_properties:
            randomize_value = self.rng.uniform(size=3)
            mass = np.interp(randomize_value[0], [-1., 1.], [0.5, 5.])
            friction = np.interp(randomize_value[0], [-1., 1.], [0, 5.])
            stiffness = np.interp(randomize_value[0], [-1., 1.], [1e4, 1.e8])
            color = list(get_board_color(stiffness=stiffness, stiff_lower=1e4, stiff_upper=1e8))+[0]
            string_model = get_cucumber_model(kp=self.object_stiffness, soft_cfm=self.object_soft_cfm, soft_erp=self.object_soft_erp)
            self.box_model = Model("block", block_pose, file_type="string",
                                   string_model=string_model, model_id="target_block")
            self.spawner.load_models([self.box_model])
        else:
            self.box_model.set_pose(block_pose)
            self.spawner.update_model_state(self.box_model)

        self.current_board_pose = transformations.pose_euler_to_quat(block_pose)

    def randomize_block_position(self):
        rand = self.rng.random(size=6)
        rand = np.array([np.interp(rand[i], [0, 1.], self.object_workspace[i]) for i in range(6)])
        rand[3:] = np.deg2rad(rand[3:])
        self.x = rand
        pose = np.copy(self.object_initial_pose)
        pose[:2] += rand[:2]  # only x and y
        return pose

    def set_environment_conditions(self):
        if self.curriculum_learning:
            if self.progressive_cl:
                if np.average(self.episode_hist) >= self.cl_upgrade_level and self.curriculum_level < 1.0:
                    self.curriculum_level += self.curriculum_level_step
                    self.logger.info("CL difficulty UP to %s, ep: %s" %
                                     (round(self.curriculum_level, 2), self.episode_num))
                    self.episode_hist = np.array([0, 1]*10)  # set 50%
                elif np.average(self.episode_hist) <= self.cl_downgrade_level and self.curriculum_level > self.curriculum_level_step:
                    self.curriculum_level -= self.curriculum_level_step
                    self.logger.info("CL difficulty DOWN to %s, ep: %s" %
                                     (round(self.curriculum_level, 2), self.episode_num))
                    self.episode_hist = np.array([0, 1]*10)  # set 50%
                self.curriculum_level = np.clip(self.curriculum_level, 0, 1)
                curriculum_level = self.curriculum_level
                self.logger.info("current CL difficulty: %s, %s, ep: %s" % (round(curriculum_level, 2),
                                 np.round(np.average(self.episode_hist), 2), self.episode_num))
            else:
                max_episodes = 200
                num_eps = self.episode_num + self.cumulative_episode_num  # current training session + previous ones
                curriculum_level = min((num_eps/max_episodes), 1.0)
                self.logger.info(
                    "current CL difficulty: %s, %s, ep: %s" %
                    (round(curriculum_level, 2),
                     np.average(self.episode_hist),
                     self.episode_num))
            self.difficulty_ratio = copy(curriculum_level)
            self.mu_range = get_cl_range(self.max_mu_range, curriculum_level)
            self.mu2_range = get_cl_range(self.max_mu_range, curriculum_level)
            self.stiffness_range = get_cl_range(self.max_stiffness_range, curriculum_level)
            self.scale_range = get_cl_range(self.max_scale_range, curriculum_level)
            self.current_object_workspace = [
                [-max(self.max_object_workspace[i] * curriculum_level, self.min_object_workspace[i]),
                 max(self.max_object_workspace[i] * curriculum_level, self.min_object_workspace[i])] for i in range(6)]
            self.current_object_workspace[2] = [max(self.max_object_workspace[2]
                                                    * curriculum_level, self.min_object_workspace[2]), 0]
            self.position_threshold_cl = 0.005 + (1-curriculum_level) * self.max_position_threshold
        else:
            self.mu_range = self.max_mu_range
            self.mu2_range = self.max_mu_range
            self.stiffness_range = self.max_stiffness_range
            self.scale_range = self.max_scale_range
            self.current_object_workspace = self.object_workspace
            self.position_threshold_cl = self.position_threshold

    def _is_done(self, observations):
        pose_error = np.abs(observations[:6]*self.max_distance)
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            self.logger.error("Collision!")

        if self.termination_on_negative_reward:
            if self.reward_based_on_cl:
                if self.cumulated_episode_reward <= self.termination_reward_threshold*self.difficulty_ratio:
                    rospy.loginfo("Fail on reward: %s" % (pose_error))
                    self.success_end = False
                    return True
            if self.cumulated_episode_reward <= self.termination_reward_threshold:
                rospy.loginfo("Fail on reward: %s" % (pose_error))
                self.success_end = False
                return True

        if self.two_steps:
            if self.stage == 0:
                position_reached = np.all(pose_error[:3] < 0.02)
            if self.stage == 1:
                position_reached = np.all(pose_error[:3] < self.position_threshold_cl)
            if position_reached and self.stage == 0:
                rospy.loginfo("First stage reached")
                self.goal_offset = 0.0
                self.stage = 1
                return False
            if position_reached and self.stage == 1:
                self.logger.info("goal reached: %s" % np.round(pose_error[:3], 4))
                self.success_end = True
            if self.step_count == self.steps_per_episode-1:
                self.logger.error("Fail!: %s" % np.round(pose_error[:3], 4))
            return (position_reached and self.stage == 1) \
                or self.action_result == FORCE_TORQUE_EXCEEDED  # Stop on collision
        else:
            position_reached = np.all(pose_error[:3] < self.position_threshold_cl)
            if self.real_robot and self.step_count % 100 == 0:
                print("----------- step:", self.step_count)
                print("dist error:", np.round(pose_error[:3], 4).tolist())
                print("motion act:", np.round(self.last_actions[:3].tolist(), 5).tolist(),
                      "\nposPID act:", np.round(self.last_actions[6:9].tolist(), 5).tolist(),
                      "\nforcePID act:", np.round(self.last_actions[12:15].tolist(), 5).tolist(),
                      "\nalpha act:", np.round(self.last_actions[18:21].tolist(), 5).tolist())
            if position_reached:
                self.logger.info("goal reached: %s" % np.round(pose_error[:3], 4))
                self.success_end = True
                if self.real_robot:
                    xc = transformations.transform_pose(
                        self.ur3e_arm.end_effector(), [0, 0, 0.013, 0, 0, 0], rotated_frame=True)
                    reset_time = 5.0
                    # print("paused")
                    # input()
                    self.ur3e_arm.set_target_pose(pose=xc, t=reset_time, wait=True)
            if self.step_count == self.steps_per_episode-1:
                self.logger.error("Fail!: %s" % np.round(pose_error[:3], 4))
            return (position_reached) \
                or self.action_result == FORCE_TORQUE_EXCEEDED  # Stop on collision

    def _get_info(self):
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            self.logger.error("Collision!")
            return "collision"
        return {}

    def _set_action(self, action):
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
