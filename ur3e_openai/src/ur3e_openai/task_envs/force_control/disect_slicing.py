
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED
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
        self.normal_randomization = rospy.get_param(prefix + "/normal_randomization", True)
        self.basic_randomization = rospy.get_param(prefix + "/basic_randomization", False)
        self.random_type = rospy.get_param(prefix + "/random_type", "uniform")
        self.cl_upgrade_level = rospy.get_param(prefix + "/cl_upgrade_level", 0.8)
        self.cl_downgrade_level = rospy.get_param(prefix + "/cl_downgrade_level", 0.2)
        print(">>>>> ", self.random_type, self.curriculum_learning, self.progressive_cl, self.reward_based_on_cl, " <<<<<<")

        self.reset_motion = rospy.get_param(prefix + "/reset_motion", [-0.05, 0, 0.035, 0, 0, 0])

        self.successes_threshold = rospy.get_param(prefix + "/successes_threshold", 0)

    def _set_init_pose(self):
        self.start = timeit.default_timer()
        
        self.success_counter = 0

        # Update target pose if needed
        self.update_target_pose()

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
        # Handling the desync message
        self.desync_service()
        # Handling the reset message
        reset_mode = "random"
        config_file_path = "abc"
        viz = False
        self.reset_service(reset_mode, config_file_path, viz)

    def _is_done(self, observations):
        pose_error = np.abs(observations[:len(self.target_dims)]*self.max_distance)

        collision = self.action_result == FORCE_TORQUE_EXCEEDED
        self.goal_reached = np.all(pose_error < self.goal_threshold)
        fail_on_reward = self.termination_on_negative_reward
        out_of_workspace = np.any(pose_error > self.workspace_limit)

        if out_of_workspace:
            self.logger.error("Out of workspace, failed: %s" % np.round(pose_error, 4))

        # If the end effector remains on the target pose for several steps. Then terminate the episode
        if self.goal_reached:
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

        elif self.goal_reached and self.success_counter > self.successes_threshold:
            self.controller.stop()
            self.logger.green("goal reached: %s" % np.round(pose_error[:3], 4))
            print("time after pause", timeit.default_timer()-self.start)
            if not self.done_once:
                self.success_end = True
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
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
