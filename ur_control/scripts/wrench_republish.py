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

from ur_control.compliant_controller import CompliantController

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
    robot_urdf = "ur3e"
    rospackage = None

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
        rospackage = ""
    
    global arm
    arm = CompliantController(ft_sensor=True,
              namespace=ns, 
              joint_names_prefix=joints_prefix, 
              robot_urdf=robot_urdf, 
              robot_urdf_package=rospackage, 
              relative_to_ee=args.relative)

    rospy.sleep(0.5)
    arm.set_wrench_offset(override=args.zero)

    offset_cnt = 0

    while not rospy.is_shutdown():
        arm.publish_wrench()

        if offset_cnt > 100:
            arm.set_wrench_offset(False)
            offset_cnt = 0
        offset_cnt += 1


main()
