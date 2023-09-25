
#!/usr/bin/env python

import copy
import rospy
import numpy as np

from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import spalg, transformations, conversions
from ur_control.constants import FORCE_TORQUE_EXCEEDED
import threading

from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

from disect.cutting import load_settings, CuttingSim, Parameter


def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


def constraint_pose(pose):
    euler_pose = transformations.pose_quaternion_to_euler(pose)
    # constraint rotation on y and z
    euler_pose[4] = 0
    euler_pose[5] = 0
    return transformations.pose_euler_to_quat(euler_pose)


class UR3eSlicingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        self.load_disect = rospy.ServiceProxy('/disect/load', Empty)
        self.reset_disect = rospy.ServiceProxy('/disect/reset', Empty)
        self.disect_step_sim = rospy.ServiceProxy('/disect/step_simulation', Empty)

        self.load_disect.wait_for_service()
        self.reset_disect.wait_for_service()
        self.load_disect()

        # Publish pose to another topic
        msg = conversions.to_pose_stamped(self.ur3e_arm.base_link, [0, 0, 0, 0, 0, 0, 1])
        robot_base_to_disect = self.tf_listener.transformPose('cutting_board_disect', msg)
        self.transform_pose = transformations.pose_to_transform(conversions.from_pose_to_list(robot_base_to_disect.pose))
        self.pub_odom = rospy.Publisher('/disect/knife/odom', Odometry, queue_size=10)

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

        self.new_sim = rospy.get_param(prefix + "/new_sim", True)
        self.viz = rospy.get_param(prefix + "/viz", True)

        # Cut completion
        self.cut_completion = 0.

        self.goal_reached = False

        self.mu1 = rospy.get_param(prefix + "/mu1", 0)
        self.mu2 = rospy.get_param(prefix + "/mu2", 0.5)
        self.mu3 = rospy.get_param(prefix + "/mu3", 1)

        self.spawn_interval = 5  # 10
        self.cumulated_dist = 0
        self.cumulated_force = 0
        self.cumulated_jerk = 0
        self.cumulated_vel = 0
        self.cumulated_reward_details = np.zeros(7)
        self.episode_count = 0

    def _set_init_pose(self):
        self.goal_reached = False
        self.success_counter = 0

        # Update target pose if needed
        self.current_target_pose = rospy.get_param("ur3e_gym/target_pose", False)

        def reset_pose():
            # Go to initial pose
            reset_motion = self.reset_motion
            # reset_motion[1] += self.np_random.uniform(low=np.deg2rad(-0.02), high=np.deg2rad(0.02))
            # reset_motion[2] += self.np_random.uniform(low=np.deg2rad(-0.01), high=np.deg2rad(0.01))
            # reset_motion[4] = self.np_random.uniform(low=np.deg2rad(-10), high=np.deg2rad(10))
            initial_pose = transformations.transform_pose(self.current_target_pose, reset_motion, rotated_frame=False)
            self.ur3e_arm.set_target_pose(pose=initial_pose, wait=True, t=self.reset_time)

        t1 = threading.Thread(target=reset_pose)
        t2 = threading.Thread(target=self.update_scene)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.reset_disect()

        disect_knife_pose, disect_knife_twist = self.compute_disect_knife_pose()
        self.publish_odom(disect_knife_pose, disect_knife_twist, update_pose=True)
        self.disect_step_sim()

        self.ur3e_arm.zero_ft_sensor()
        self.controller.reset()
        self.controller.start()

    def _is_done(self, observations):
        pose_error = np.abs(observations[:len(self.target_dims)]*self.max_distance)

        collision = self.action_result == FORCE_TORQUE_EXCEEDED
        position_goal_reached = np.all(pose_error < self.goal_threshold)
        fail_on_reward = self.termination_on_negative_reward
        self.out_of_workspace = np.any(pose_error > self.workspace_limit)

        # If the end effector remains on the target pose for several steps. Then terminate the episode
        if position_goal_reached:
            self.success_counter += 1
        # else:
        #     self.success_counter = 0

        if self.step_count == self.steps_per_episode-1:
            self.logger.error("Max steps x episode reached, failed: %s" % np.round(pose_error, 4))
            self.robot_connection.unpause()
            self.controller.stop()
            self.robot_connection.pause()

        if collision:
            self.logger.error("Collision! pose: %s" % (pose_error))

        elif fail_on_reward:
            if self.reward_based_on_cl:
                if self.cumulated_episode_reward <= self.termination_reward_threshold*self.difficulty_ratio:
                    rospy.loginfo("Fail on reward: %s" % (pose_error))
            if self.cumulated_episode_reward <= self.termination_reward_threshold:
                rospy.loginfo("Fail on reward: %s" % (pose_error))

        elif position_goal_reached and self.success_counter > self.successes_threshold:
            self.goal_reached = True
            self.logger.green("goal reached: %s" % np.round(pose_error[:3], 4))

        done = self.goal_reached or collision or fail_on_reward or self.out_of_workspace

        if done:
            self.robot_connection.unpause()
            self.controller.stop()
            self.robot_connection.pause()

        return done

    def _get_info(self, obs):
        return {"success": self.goal_reached,
                "collision": self.action_result == FORCE_TORQUE_EXCEEDED,
                "dist": self.cumulated_dist,
                "force": self.cumulated_force,
                "jerk": self.cumulated_jerk,
                "vel": self.cumulated_vel,
                "cumulated_reward_details": self.cumulated_reward_details}

    def _set_action(self, action):
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)

    def compute_disect_knife_pose(self):
        knife_pose = self.ur3e_arm.end_effector(tip_link='b_bot_knife_sim')
        disect_knife_pose = transformations.apply_transformation(knife_pose, self.transform_pose)
        disect_knife_pose[1] -= 0.0043
        return disect_knife_pose, np.zeros(6)

    def publish_odom(self, pose, twist, update_pose=False):
        msg = Odometry()
        msg.header.frame_id = "update_pose" if update_pose else ""
        msg.pose.pose = conversions.to_pose(pose)
        msg.twist.twist.linear = conversions.to_vector3(twist[:3])
        msg.twist.twist.angular = conversions.to_vector3(twist[3:])
        self.pub_odom.publish(msg)
