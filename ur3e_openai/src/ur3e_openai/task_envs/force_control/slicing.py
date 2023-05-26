
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.robot_envs.utils import get_board_color
from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import spalg, transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_gazebo.basic_models import get_button_model
from ur_gazebo.model import Model

import threading


def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


class UR3eSlicingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        if not self.real_robot:
            string_model = get_button_model(spring_stiffness=self.object_stiffness, damping=self.object_damping, friction=self.object_friction,
                                            base_mass=1., button_mass=0.1, color=[1, 0, 0, 0], kp=self.object_kp, kd=self.object_kd, max_vel=100.0)
            self.box_model = Model("block", self.object_initial_pose, file_type="string",
                                   string_model=string_model, model_id="target_block", reference_frame="o2ac_ground")

    def __load_env_params(self):
        prefix = "ur3e_gym"

        # Gazebo spawner parameters
        self.randomize_object_properties = rospy.get_param(prefix + "/randomize_object_properties", False)
        self.object_initial_pose = rospy.get_param(prefix + "/object_initial_pose", [])
        self.object_stiffness = rospy.get_param(prefix + "/object_stiffness", 400)
        self.object_damping = rospy.get_param(prefix + "/object_damping", 0.0)
        self.object_kp = rospy.get_param(prefix + "/object_kp", 1e8)
        self.object_kd = rospy.get_param(prefix + "/object_kd", 1)
        self.object_friction = rospy.get_param(prefix + "/object_friction", 1.0)

        self.uncertainty_error = rospy.get_param(prefix + "/uncertainty_error", False)
        self.uncertainty_error_max_range = rospy.get_param(prefix + "/uncertainty_error_max_range", [0, 0, 0, 0, 0, 0])

        self.max_stiffness_range = rospy.get_param(prefix + "/max_stiffness_range", [5e5, 1e6])
        self.max_damping_range = rospy.get_param(prefix + "/max_damping_range", [0, 2])
        self.max_kp_range = rospy.get_param(prefix + "/max_kp_range", [4e5, 1e6])
        self.max_kd_range = rospy.get_param(prefix + "/max_kd_range", [1, 2])
        self.max_friction_range = rospy.get_param(prefix + "/max_friction_range", [0, 1])

        self.update_target = rospy.get_param(prefix + "/update_target", False)

        self.reset_motion = rospy.get_param(prefix + "/reset_motion", [-0.05, 0, 0.035, 0, 0, 0])

        # How often to generate a new model, number of episodes
        self.normal_randomization = rospy.get_param(prefix + "/normal_randomization", True)
        self.basic_randomization = rospy.get_param(prefix + "/basic_randomization", False)
        self.random_type = rospy.get_param(prefix + "/random_type", "uniform")
        self.cl_upgrade_level = rospy.get_param(prefix + "/cl_upgrade_level", 0.8)
        self.cl_downgrade_level = rospy.get_param(prefix + "/cl_downgrade_level", 0.2)
        print(">>>>> ", self.random_type, self.curriculum_learning,
              self.progressive_cl, self.reward_based_on_cl, " <<<<<<")

        self.successes_threshold = rospy.get_param(prefix + "/successes_threshold", 0)

    def _set_init_pose(self):
        self.success_counter = 0
        self.goal_reached = False

        # Update target pose if needed
        self.update_target_pose()

        # For real robot, do nothing for reset, reset somewhere else
        if rospy.get_param("ur3e_gym/update_initial_conditions", True):
            def reset_pose():
                # Go to initial pose
                initial_pose = transformations.transform_pose(self.current_target_pose, self.reset_motion, rotated_frame=False)
                self.ur3e_arm.set_target_pose(pose=initial_pose, wait=True, t=self.reset_time)

            t1 = threading.Thread(target=reset_pose)
            t2 = threading.Thread(target=self.update_scene)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        self.ur3e_arm.zero_ft_sensor()
        self.controller.reset()
        self.controller.start()

    def update_scene(self):
        if self.real_robot:
            return

        self.stage = 0
        block_pose = self.object_initial_pose
        # Domain randomization:
        if self.randomize_object_properties:
            randomize_value = self.np_random.uniform(low=0.0, high=1.0, size=4)

            stiffness = np.interp(randomize_value[0], [0., 1.], self.max_stiffness_range)
            damping = np.interp(randomize_value[1], [0., 1.], self.max_damping_range)
            friction = np.interp(randomize_value[1], [0., 1.], self.max_friction_range)
            kp = np.interp(randomize_value[2], [0., 1.], self.max_kp_range)
            kd = np.interp(randomize_value[3], [0., 1.], self.max_kd_range)
            color = list(get_board_color(stiffness=stiffness,
                         stiff_lower=self.max_stiffness_range[0], stiff_upper=self.max_stiffness_range[1]))
            color[3] = 0.1
            string_model = get_button_model(spring_stiffness=stiffness, damping=damping, friction=friction,
                                            base_mass=1., button_mass=0.1, color=color, kp=kp, kd=kd, max_vel=100.0)
            self.box_model = Model("block", block_pose, file_type="string", string_model=string_model, model_id="target_block")
            self.spawner.reset_model(self.box_model)
        else:
            self.box_model.set_pose(block_pose)
            self.spawner.update_model_state(self.box_model)

        self.current_board_pose = transformations.pose_euler_to_quat(block_pose)

    def _is_done(self, observations):
        pose_error = np.abs(observations[:len(self.target_dims)]*self.max_distance)

        collision = self.action_result == FORCE_TORQUE_EXCEEDED
        position_goal_reached = np.all(pose_error < self.goal_threshold)
        fail_on_reward = self.termination_on_negative_reward
        out_of_workspace = np.any(pose_error > self.workspace_limit)

        if out_of_workspace:
            self.logger.error("Out of workspace, failed: %s" % np.round(pose_error, 4))

        # If the end effector remains on the target pose for several steps. Then terminate the episode
        if position_goal_reached:
            self.success_counter += 1
        else:
            self.success_counter = 0

        if self.step_count == self.steps_per_episode-1:
            self.logger.error("Max steps x episode reached, failed: %s" % np.round(pose_error, 4))

        if collision:
            self.logger.error("Collision!")

        elif fail_on_reward:
            if self.reward_based_on_cl:
                if self.cumulated_episode_reward <= self.termination_reward_threshold*self.difficulty_ratio:
                    rospy.loginfo("Fail on reward: %s" % (pose_error))
            if self.cumulated_episode_reward <= self.termination_reward_threshold:
                rospy.loginfo("Fail on reward: %s" % (pose_error))

        elif position_goal_reached and self.success_counter > self.successes_threshold:
            self.goal_reached = True
            self.controller.stop()
            self.logger.green("goal reached: %s" % np.round(pose_error[:3], 4))
            
            if self.real_robot:
                xc = transformations.transform_pose(self.ur3e_arm.end_effector(), [0, 0, 0.013, 0, 0, 0], rotated_frame=True)
                reset_time = 5.0
                self.ur3e_arm.set_target_pose(pose=xc, t=reset_time, wait=True)

        done = self.goal_reached or collision or fail_on_reward or out_of_workspace

        if done:
            self.controller.stop()

        return done

    def _get_info(self):
        return {"success": self.goal_reached,
                "collision": self.action_result == FORCE_TORQUE_EXCEEDED}

    def _set_action(self, action):
        self.last_actions = action.copy()
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
