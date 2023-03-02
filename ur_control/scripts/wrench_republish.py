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

from geometry_msgs.msg import WrenchStamped
from ur_control import conversions, spalg
from ur_control.arm import Arm



def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm', default=None)
    parser.add_argument('--zero', action='store_true', help='reset ft at start')
    parser.add_argument('--relative', action='store_true', help='FT relative to EE')

    args = parser.parse_args()

    rospy.init_node('ur3e_wrench_republisher')

    ns = ''
    joints_prefix = None

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
    
    global arm
    arm = Arm(ft_topic="wrench",
              namespace=ns, 
              joint_names_prefix=joints_prefix, 
              ee_link="gripper_tip_link")

    r = rospy.Rate(500)

    pub = rospy.Publisher(ns + "/wrench/knife", WrenchStamped, queue_size=10)

    while not rospy.is_shutdown():
        sensor_wrench = arm.get_ee_wrench()
        knife_wrench = sensor_wrench.copy()
        knife_wrench[:3] += spalg.sensor_torque_to_tcp_force(tcp_position=[0.0, -0.158, 0.0], sensor_torques=sensor_wrench[3:])
        knife_wrench[3:] = np.zeros(3)
        
        msg = WrenchStamped()
        msg.wrench = conversions.to_wrench(knife_wrench)

        pub.publish(msg)
        
        r.sleep()


main()
