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
from ur_control import conversions, utils

from sensor_msgs.msg import Imu


class ImuFake(object):

    def __init__(self, topic, namespace="", frequency=500):

        self.ns = namespace

        self.topic = utils.solve_namespace(namespace + "/" + topic)

        # Publisher to outward topic
        self.pub = rospy.Publisher(self.topic, Imu, queue_size=10)

        prefix = "" if not namespace else namespace + "_"
        base_link = "base_link"

        gravity_on_base_link = np.array([0, 0, 9.81])

        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            msg = Imu()
            msg.header.frame_id = prefix + base_link
            msg.header.stamp = rospy.Time.now()
            msg.linear_acceleration = conversions.to_vector3(gravity_on_base_link)
            self.pub.publish(msg)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Filter FT signal')
    parser.add_argument('-ns', '--namespace', type=str, help='Namespace', required=False, default="")
    
    args, unknown = parser.parse_known_args()

    rospy.init_node('imu_fake')

    ft_sensor = ImuFake(topic="imu", namespace=args.namespace)
    
    rospy.spin()


main()
