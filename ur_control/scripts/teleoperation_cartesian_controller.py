#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2023 Cristian Beltran
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
# Author: Cristian Beltran

import rospy
import threading
import sys
import signal

from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import Joy

from ur_control import constants
from ur_control.fzi_cartesian_compliance_controller import CompliantController

from vive_tracking_ros.teleoperation_base import TeleoperationBase


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Teleoperation(TeleoperationBase):
    """ Implementation of Teleoperation with Vive controllers using the FZI Cartesian Compliance Controllers """

    def __init__(self) -> None:
        super(Teleoperation, self).__init__()

        self.initial_configuration = rospy.get_param("~initial_configuration")

        self.control_frequency = rospy.get_param('~control_frequency', default=100)
        self.rate = rospy.Rate(self.control_frequency)

        self.incoming_command_timeout = rospy.get_param('~incoming_command_timeout', default=0.1)
        self.controller_stiffness = rospy.get_param('~controller_stiffness', [2000., 2000., 2000., 200., 50., 50.])

        joint_names_prefix = self.robot_ns + "_" if self.robot_ns else ""
        no_prefix_end_effector = self.end_effector.replace(self.robot_ns+"_", "")

        self.robot_arm = CompliantController(namespace=self.robot_ns,
                                             joint_names_prefix=joint_names_prefix,
                                             ee_link=no_prefix_end_effector,
                                             ft_topic='wrench',
                                             gripper=constants.ROBOTIQ_GRIPPER
                                             )

        self.reset_robot_pose_request = False

        self.last_msg_mutex = threading.Lock()

        self.last_msg_time = rospy.get_time()

        rospy.Subscriber('%s/%s/target_frame' % (self.robot_ns, constants.CARTESIAN_COMPLIANCE_CONTROLLER), PoseStamped, self.target_pose_cb)
        rospy.Subscriber('%s/%s/target_wrench' % (self.robot_ns, constants.CARTESIAN_COMPLIANCE_CONTROLLER), WrenchStamped, self.target_wrench_cb)

    # Overwriting
    def joy_cb(self, data):
        super(Teleoperation, self).joy_cb(data)

        menu_button = data.buttons[0]
        touchpad_button = data.buttons[1]
        grip_button = data.buttons[3]

        if touchpad_button:
            rospy.loginfo("=== Resetting robot pose ===")
            self.reset_robot_pose_request = True
            rospy.sleep(1.0)
        if grip_button:
            gripper_state = 'open' if self.robot_arm.gripper.get_position() > 0.08 else 'close'
            if gripper_state == 'open':
                rospy.loginfo("=== Closing Gripper ===")
                self.robot_arm.gripper.close(wait=False)
            else:
                rospy.loginfo("=== Opening Gripper ===")
                self.robot_arm.gripper.open(wait=False)
            rospy.sleep(1.0)
        if menu_button:
            rospy.loginfo("=== Zeroing FT sensor ===")
            self.robot_arm.zero_ft_sensor()
            rospy.sleep(1.0)

    def target_pose_cb(self, _):
        with self.last_msg_mutex:
            self.last_msg_time = rospy.get_time()

    def target_wrench_cb(self, _):
        with self.last_msg_mutex:
            self.last_msg_time = rospy.get_time()

    def reset_robot_pose(self):
        self.robot_arm.activate_joint_trajectory_controller()
        self.robot_arm.set_joint_positions(self.initial_configuration, t=5, wait=True)
        self.robot_arm.zero_ft_sensor()
        self.center_target_pose()

    def run(self):
        rospy.loginfo("=== Moving to initial configuration ===")

        self.robot_arm.set_control_mode(mode="spring-mass-damper")
        self.robot_arm.update_stiffness(self.controller_stiffness)
        self.robot_arm.update_pd_gains(p_gains=[0.05, 0.05, 0.05, 1.0, 1.0, 1.0])
        self.robot_arm.set_solver_parameters(error_scale=0.6, iterations=1)

        compliance_controller_activated = False

        self.reset_robot_pose()

        rospy.loginfo("=== Ready for teleoperation ===")
        while not rospy.is_shutdown():
            if self.reset_robot_pose_request:
                self.set_tracking(enable=False)
                self.reset_robot_pose()
                self.reset_robot_pose_request = False
                compliance_controller_activated = False

            enable_compliance = False
            with self.last_msg_mutex:
                # Stop if there is no more incoming commands in more that some time
                if (rospy.get_time() - self.last_msg_time) > self.incoming_command_timeout:
                    enable_compliance = False
                else:
                    enable_compliance = True

            if enable_compliance and not compliance_controller_activated:
                self.robot_arm.activate_cartesian_controller()
                compliance_controller_activated = True

            if not enable_compliance and compliance_controller_activated:
                self.robot_arm.activate_joint_trajectory_controller()
                compliance_controller_activated = False

            self.rate.sleep()


if __name__ == "__main__":
    teleop = Teleoperation()
    teleop.run()
