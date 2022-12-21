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
from ur_control import utils, arm, conversions

from geometry_msgs.msg import WrenchStamped


class FTsensor(object):

    def __init__(self, in_topic, out_topic=None,
                 timeout=3.0):
        robot_name = "b_bot"
        tcp_link = "knife_center"
        self.arm = arm.Arm(ft_sensor=True, namespace=robot_name,
                           joint_names_prefix=robot_name+'_', robot_urdf=robot_name,
                           robot_urdf_package='o2ac_scene_description',
                           ee_link=tcp_link)

        self.in_topic = utils.solve_namespace(in_topic)
        if out_topic:
            self.out_topic = utils.solve_namespace(out_topic)
        else:
            self.out_topic = self.in_topic + 'wrench'

        rospy.loginfo("Publishing filtered FT to %s" % self.out_topic)

        # Publisher to outward topic
        self.pub = rospy.Publisher(self.out_topic, WrenchStamped, queue_size=10)

        # Subscribe to incoming topic
        rospy.Subscriber(self.in_topic, WrenchStamped, self.cb_raw)

        rospy.loginfo('Pub successfully initialized')

    def cb_raw(self, msg):
        if rospy.is_shutdown():
            return
        msg_out = WrenchStamped()
        msg_out.wrench = conversions.to_wrench(self.arm.get_ee_wrench())
        self.pub.publish(msg_out)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Filter FT signal')
    parser.add_argument('-t', '--ft_topic', type=str, help='FT sensor data topic', required=True)

    args, unknown = parser.parse_known_args()

    rospy.init_node('ft_filter')

    ft_sensor = FTsensor(in_topic=args.ft_topic)

    rospy.spin()


main()
