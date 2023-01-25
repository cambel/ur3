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


def move_joints():
    q = [1.0639, -1.6225, 1.9346, -1.8819, -1.5647, -0.5853]
    arm.set_joint_positions(q, t=5)


def move_cartesian():

    arm.set_position_control_mode(False)

    ee = arm.end_effector()

    p1 = ee.copy()
    p1[2] += 0.005

    p2 = p1.copy()
    p2[2] += 0.005

    trajectory = np.stack((p1, p2))
    target_force = np.zeros(6)

    arm.execute_compliance_control(trajectory, target_force=target_force,
                                   max_force_torque=[50., 50., 50., 5., 5., 5.], duration=5)

    print("EE change", ee - arm.end_effector())


def move_force():
    """ Linear push. Move until the target force is felt and stop. """
    arm.zero_ft_sensor()

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    pid_gains = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    arm.update_pid_gains(pid_gains)

    ee = arm.end_effector()

    target_force = np.zeros(6)
    target_force[2] = -5

    res = arm.execute_compliance_control(ee, target_force=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=10,
                                         stop_on_target_force=True)
    print(res)
    print("EE change", ee - arm.end_effector())


def admittance_control():
    """ Spring-mass-damper force control """
    arm.set_control_mode(mode="spring-mass-damper")

    ee = arm.end_effector()
    target_force = np.zeros(6)
    arm.execute_compliance_control(ee, target_force=target_force,
                                   max_force_torque=[50., 50., 50., 5., 5., 5.], duration=10,
                                   stop_on_target_force=False)


def free_drive():
    arm.zero_ft_sensor()

    selection_matrix = [0, 0, 0, 0, 0, 0]
    arm.update_selection_matrix(selection_matrix)

    pid_gains = [0.05, 0.05, 0.05, 1.0, 1.0, 1.0]
    arm.update_pid_gains(pid_gains)

    ee = arm.end_effector()

    target_force = np.zeros(6)

    res = arm.execute_compliance_control(ee, target_force=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=15,
                                         stop_on_target_force=False)
    print(res)
    print("EE change", ee - arm.end_effector())


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move_joints', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-mc', '--move_cartesian', action='store_true',
                        help='move to cartesian configuration')
    parser.add_argument('-mf', '--move_force', action='store_true',
                        help='move towards target force')
    parser.add_argument('-fd', '--free_drive', action='store_true',
                        help='move the robot freely')
    parser.add_argument('-hfc', '--hand_frame_control', action='store_true',
                        help='move towards target force using hand frame of reference')
    parser.add_argument('-a', '--admittance', action='store_true',
                        help='Spring-mass-damper force control demo')
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

    if args.move_joints:
        move_joints()
    if args.move_cartesian:
        move_cartesian()
    if args.move_force:
        move_force()
    if args.admittance:
        admittance_control()
    if args.free_drive:
        free_drive()
    if args.hand_frame_control:
        move_hand_frame_control()


if __name__ == "__main__":
    main()