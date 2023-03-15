
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.robot_envs.utils import get_board_color
from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import spalg, transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND
from ur_gazebo.basic_models import get_box_model, get_button_model, get_cucumber_model
from ur_gazebo.model import Model

import threading


def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


class UR3ePokingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

    def __load_env_params(self):
        prefix = "ur3e_gym"

        self.position_threshold_cl = self.position_threshold
        self.successes_threshold = rospy.get_param(prefix + "/successes_threshold", 0)

    def _set_init_pose(self):
        """
        Define initial pose if random pose is desired.
        Otherwise, use manually defined initial pose.
        Then move the Robot to its initial pose
        """
        self.controller.stop()

        # Manually define a joint initial configuration
        qc = self.init_q
        self.ur3e_arm.set_joint_positions(position=qc, wait=True, t=self.reset_time)

        self.current_target_pose = rospy.get_param("ur3e_gym/target_pose", False)

        self.ur3e_arm.zero_ft_sensor()
        self.controller.start()

    def _is_done(self, observations):
        # Check for excessive contact force
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            self.logger.error("Collision! Aborting")
            self.controller.stop()
            return True

        pose_error = np.abs(observations[:6]*self.max_distance)

        # Check how close we are to the target pose
        position_reached = np.all(pose_error[:3] < self.position_threshold_cl)

        # If the end effector remains on the target pose for several steps. Then terminate the episode
        if position_reached:
            self.success_counter += 1
        else:
            self.success_counter = 0

        # After 0.1 seconds, terminate the episode
        # if self.success_counter > (0.05 / self.agent_control_dt):
        if self.success_counter > self.successes_threshold:
            self.logger.info("goal reached: %s" % np.round(pose_error[:3], 4))
            self.success_end = True
            if self.real_robot:
                xc = transformations.transform_pose(self.ur3e_arm.end_effector(), [0, 0, 0.013, 0, 0, 0], rotated_frame=True)
                reset_time = 5.0
                self.ur3e_arm.set_target_pose(pose=xc, t=reset_time, wait=True)

            self.controller.stop()
            return True

        # Check whether we took every available step for this episode
        if self.step_count == self.steps_per_episode-1:
            self.logger.error("Fail!: %s" % np.round(pose_error[:3], 4))

            self.controller.stop()
            return True

        return False

    def _get_info(self):
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            return "collision"
        return {}

    def _set_action(self, action):
        # print(np.round(self.current_target_pose[:3], 4).tolist())
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
