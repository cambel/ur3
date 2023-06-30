#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from vive_tracking_ros.msg import ViveControllerFeedback
import tf2_ros
import sys

from ur_control import conversions, spalg, transformations


class Converter:
    """ Convert Twist messages to PoseStamped

    Use this node to integrate twist messages into a moving target pose in
    Cartesian space.  An initial TF lookup assures that the target pose always
    starts at the robot's end-effector.
    """

    def __init__(self):
        rospy.init_node('converter', anonymous=False)

        self.robot_ns = rospy.get_param('~robot_namespace', default="")
        self.robot_frame_id = rospy.get_param('~robot_frame_id', default="base_link")
        self.end_effector = rospy.get_param(self.robot_ns + '/cartesian_compliance_controller/end_effector_link', default="tool0")

        self.vive_base_frame = rospy.get_param('~vive_frame_id', default="world")
        self.controller_name = rospy.get_param('~controller_name', default="right_controller")

        pose_topic = rospy.get_param('~pose_topic', default="my_pose")
        wrench_topic = rospy.get_param('~wrench_topic', default="/wrench")
        haptic_feedback_topic = rospy.get_param('~haptic_feedback_topic', default="/vive/set_feedback")
        twist_topic = '/vive/' + self.controller_name + '/twist'
        joy_topic = '/vive/' + self.controller_name + '/joy'

        # Limit the displacement to the play area
        # Todo limit rotation
        self.play_area = rospy.get_param('~play_area', [0.05, 0.05, 0.05, 15, 15, 15])
        self.play_area[3:] = np.deg2rad(self.play_area[3:])
        # Limit contact interaction
        self.max_force_torque = rospy.get_param('~max_contact_force_torque', default=[50., 50., 50., 5., 5., 5.])

        self.scale_velocities = rospy.get_param('~scale_velocities', [1., 1., 1., 1., 1., 1.])
        self.scale_velocities = np.clip(self.scale_velocities, 0.0, 1.0)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.vive_to_robot_rotation = conversions.from_quaternion(self.get_transformation(source=self.vive_base_frame, target=self.robot_frame_id).transform.rotation)

        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])
        self.center_position = np.zeros(3)
        self.center_orientation = np.array([0, 0, 0, 1])

        self.enable_tracking = False

        # Start where we are
        if not self.center_target_pose():
            sys.exit(0)

        self.target_pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=3)
        self.twist_sub = rospy.Subscriber(twist_topic, Twist, self.twist_cb)
        self.joy_inputs_sub = rospy.Subscriber(joy_topic, Joy, self.joy_cb, queue_size=1)

        self.haptic_feedback_rate = rospy.Rate(40)
        self.haptic_feedback_pub = rospy.Publisher(haptic_feedback_topic, ViveControllerFeedback, queue_size=3)
        self.wrench_sub = rospy.Subscriber(wrench_topic, WrenchStamped, self.wrench_cb, queue_size=1)


    def get_transformation(self, source, target):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target, source_frame=source, time=rospy.Time(0), timeout=rospy.Duration(5))

        except (tf2_ros.InvalidArgumentException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return False

    def center_target_pose(self):
        start = self.get_transformation(source=self.end_effector, target=self.robot_frame_id)

        if not start:
            return False

        self.target_position = conversions.from_point(start.transform.translation)
        self.target_orientation = conversions.from_quaternion(start.transform.rotation)

        self.center_position = self.target_position
        self.center_orientation = self.target_orientation

        self.last = rospy.get_time()

        return True

    def twist_cb(self, data: Twist):
        """ Numerically integrate twist message into a pose

        Use global self.frame_id as reference for the navigation commands.
        """

        if not self.enable_tracking:
            return

        now = rospy.get_time()
        dt = now - self.last
        self.last = now

        dt = dt

        linear_vel = conversions.from_vector3(data.linear) * self.scale_velocities[:3]
        angular_vel = conversions.from_vector3(data.angular) * self.scale_velocities[3:]

        # transform to robot base frame
        linear_vel = transformations.quaternion_rotate_vector(self.vive_to_robot_rotation, linear_vel)
        angular_vel = transformations.quaternion_rotate_vector(self.vive_to_robot_rotation, angular_vel)

        # Position update
        next_pose = self.target_position + (linear_vel * dt)

        # Orientation update
        next_orientation = transformations.integrateUnitQuaternionDMM(self.target_orientation, angular_vel, dt)

        translation = next_pose - self.center_position
        rotation = spalg.quaternions_orientation_error(next_orientation, self.center_orientation)

        # DEBUG prints
        # rospy.loginfo_throttle(1, "translation %s dt %s" % (translation, dt))
        # rospy.loginfo_throttle(1, "rotation %s dt %s" % (rotation, dt))

        if np.any(np.abs(translation) > self.play_area[:3]) or np.any(np.abs(rotation) > self.play_area[3:]):
            return
        else:
            self.target_position = next_pose
            self.target_orientation = next_orientation  # the last one is after dt passed

        self.publish_target_pose()

    def joy_cb(self, data: Joy):
        trigger_button = data.buttons[2]
        if trigger_button:

            # re-center the target pose
            if not self.center_target_pose():
                sys.exit(0)

            # Enable/Disable tracking
            self.enable_tracking = not self.enable_tracking
            if self.enable_tracking:
                rospy.loginfo("=== Tracking Enabled  ===")
            else:
                rospy.loginfo("=== Tracking Disabled ===")

            rospy.sleep(0.5)

    def wrench_cb(self, data: WrenchStamped):
        if not self.enable_tracking:
            return

        wrench = conversions.from_wrench(data.wrench)
        normalized_wrench = wrench / self.max_force_torque
        intensity = np.linalg.norm(normalized_wrench)
        
        rospy.loginfo_throttle(0.5, round(intensity, 2))

        if intensity > 0.1:
            haptic_msg = ViveControllerFeedback()
            haptic_msg.controller_name = self.controller_name
            haptic_msg.duration_microsecs = np.interp(intensity, [0.0, 2.45], [0.0, 3999.0])

            self.haptic_feedback_pub.publish(haptic_msg)

            self.haptic_feedback_rate.sleep()

        if np.any(wrench > self.max_force_torque):
            self.enable_tracking = False
            rospy.logwarn("Tracking stopped, excessive contact force detected: %s" % (np.round(wrench, 1)))

    def publish_target_pose(self):
        if not rospy.is_shutdown():
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.robot_frame_id
            msg.pose.position = conversions.to_point(self.target_position)
            msg.pose.orientation = conversions.to_quaternion(self.target_orientation)

            try:
                self.target_pose_pub.publish(msg)
            except rospy.ROSException:
                # Swallow 'publish() to closed topic' error.
                # This rarely happens on killing this node.
                pass


if __name__ == '__main__':
    conv = Converter()
    rospy.spin()
