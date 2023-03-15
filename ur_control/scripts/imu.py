#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
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

import argparse
import rospy

import numpy as np
from ur_control.constants import get_arm_joint_names
from ur_pykdl import ur_kinematics
from ur_control import transformations, utils

from sensor_msgs.msg import Imu
from ur_control.controllers import JointControllerBase


class ImuFake(object):

    def __init__(self, topic, namespace="", frequency=500):

        self.ns = namespace

        self.topic = utils.solve_namespace(namespace + "/" + topic)

        # Publisher to outward topic
        self.pub = rospy.Publisher(self.topic, Imu, queue_size=10)

        prefix = "" if not namespace else namespace + "_"
        base_link = "base_link"
        ft_sensor_link = "wrist_3_link"
        self.robot_state = JointControllerBase(namespace, timeout=1, joint_names=get_arm_joint_names(prefix))
        self.kdl = ur_kinematics(base_link=prefix + base_link, ee_link=prefix + ft_sensor_link)

        gravity_on_base_link = np.array([0, 0, -0.981])

        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            fk = self.kdl.forward(self.robot_state.get_joint_positions())
            gravity_on_ft_sensor = - transformations.quaternion_rotate_vector(fk[3:], gravity_on_base_link)
            msg = Imu()
            msg.header.frame_id = prefix + ft_sensor_link
            msg.linear_acceleration.x = gravity_on_ft_sensor[0]
            msg.linear_acceleration.y = gravity_on_ft_sensor[1]
            msg.linear_acceleration.z = gravity_on_ft_sensor[2]
            self.pub.publish(msg)
            rate.sleep()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Filter FT signal')
    parser.add_argument('-ns', '--namespace', type=str, help='Namespace', required=False, default="")
    
    args, unknown = parser.parse_known_args()

    rospy.init_node('imu_fake')

    ft_sensor = ImuFake(topic="imu", namespace=args.namespace)
    if args.zero:
        ft_sensor.update_wrench_offset()

    rospy.spin()


main()
