
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import spalg, transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_gazebo.model import Model
from o2ac_msgs.srv import resetDisect
from std_srvs.srv import Trigger

import threading
import timeit

def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


class UR3eSlicingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        if not self.real_robot:
            self.reset_service = rospy.ServiceProxy('reset_simulation', resetDisect) 
            self.desync_service = rospy.ServiceProxy('desync_simulation', Trigger)

    def __load_env_params(self):
        prefix = "ur3e_gym"

        # Gazebo spawner parameters

        # How often to generate a new model, number of episodes
        self.refresh_rate = rospy.get_param(prefix + "/refresh_rate", False)
        self.normal_randomization = rospy.get_param(prefix + "/normal_randomization", True)
        self.basic_randomization = rospy.get_param(prefix + "/basic_randomization", False)
        self.random_type = rospy.get_param(prefix + "/random_type", "uniform")
        self.cl_upgrade_level = rospy.get_param(prefix + "/cl_upgrade_level", 0.8)
        self.cl_downgrade_level = rospy.get_param(prefix + "/cl_downgrade_level", 0.2)
        print(">>>>> ", self.random_type, self.curriculum_learning, self.progressive_cl, self.reward_based_on_cl, " <<<<<<")

        self.position_threshold_cl = self.position_threshold
        self.successes_threshold = rospy.get_param(prefix + "/successes_threshold", 0)

    def _set_init_pose(self):
        if not self.real_robot:
            self.set_environment_conditions()

        self.success_counter = 0

        if rospy.get_param("ur3e_gym/update_initial_conditions", True):
            def reset_pose():
                # Go to initial pose
                initial_pose = transformations.transform_pose(self.current_target_pose, [-0.05, 0, 0.035, 0, 0, 0], rotated_frame=False)
                self.ur3e_arm.set_target_pose(pose=initial_pose, wait=True, t=self.reset_time)

                self.max_distance = spalg.translation_rotation_error(self.current_target_pose, initial_pose)
                self.max_distance = np.clip(self.max_distance, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [1, 1, 1, np.pi*2, np.pi*2, np.pi*2])

            t1 = threading.Thread(target=reset_pose)
            t2 = threading.Thread(target=self.update_scene)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            # self.update_scene()

        self.ur3e_arm.zero_ft_sensor()
        self.controller.start()

    def update_scene(self):
        self.start = timeit.default_timer()
        if self.real_robot:
            return

        block_pose = self.object_initial_pose

        self.current_board_pose = transformations.pose_euler_to_quat(block_pose)
        # Handling the desync message
        self.desync_service()
        # Handling the reset message
        reset_mode = "random"
        config_file_path = "abc"
        viz = False
        self.reset_service(reset_mode, config_file_path, viz)

    def _is_done(self, observations):
        # Check for excessive contact force
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            self.logger.error("Collision! Aborting")

            self.controller.stop()
            return True

        pose_error = np.abs(observations[:6]*self.max_distance)

        if self.termination_on_negative_reward:
            if self.cumulated_episode_reward <= self.termination_reward_threshold:
                rospy.loginfo("Fail on reward: %s" % np.round(pose_error[:3], 4))
                self.success_end = False
                self.controller.stop()
                return True

        # Check how close we are to the target pose (Only y and z, x could be anywhere and its fine)
        position_reached = np.all(pose_error[1:3] < self.position_threshold_cl)

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
            print("time after pause", timeit.default_timer()-self.start)
            return True

        # Check whether we took every available step for this episode
        if self.step_count == self.steps_per_episode-1:
            self.logger.error("Fail!: %s" % np.round(pose_error[:3], 4))

            self.controller.stop()
            return True

        return False

    def _get_info(self):
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            return {"collision": True}
        return {}

    def _set_action(self, action):
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
