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

import sys
import signal
from ur_control import utils, traj_utils, constants
from ur_control.fzi_cartesian_compliance_controller import CompliantController
import argparse
import rospy
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def move():

    arm.set_position_control_mode(False)

    ee = arm.end_effector()

    p1 = ee.copy()
    p1[2] += 0.005

    p2 = p1.copy()
    p2[2] += 0.005

    trajectory = np.stack((p1, p2))

    print(trajectory)

    arm.execute_compliance_control(trajectory, target_force=np.zeros(0),
                                   max_force_torque=[50., 50., 50., 5., 5., 5.], duration=5)

    print(arm.end_effector())

def force_control():
    """ 
        Simple example of compliance control
        selection_matrix: list[6]. define which direction is controlled by position(1.0) or force(0.0) goal. 
                          Values in between make the controller attempt to achieve both position and force goals.
    """
    pass


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force control demo')
    # parser.add_argument('--circle', action='store_true',
    #                     help='Circular rotation around a target pose')
    # parser.add_argument('--spiral', action='store_true',
    #                     help='Spiral rotation around a target pose')
    parser.add_argument('--namespace', type=str,
                        help='Namespace of arm', default=None)
    args = parser.parse_args()

    rospy.init_node('ur3e_compliance_control')

    ns = None
    joints_prefix = None
    tcp_link = 'gripper_tip_link'

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + '_'

    global arm
    arm = CompliantController(namespace=ns,
                              joint_names_prefix=joints_prefix,
                              ee_link=tcp_link,
                              ft_topic='wrench')

    if args.move:
        move()
    # if args.circle:
    #     circular_trajectory()
    # if args.spiral:
    #     spiral_trajectory()
    if args.force:
        force_control()


if __name__ == "__main__":
    main()
