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
    q = [1.3506, -1.6493, 1.9597, -1.8814, -1.5652, 1.3323]
    arm.set_joint_positions(q, t=5, wait=True)


def move_cartesian():
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(q, t=1, wait=True)

    arm.set_position_control_mode(False)
    # arm.set_control_mode(mode="spring-mass-damper")
    arm.set_control_mode(mode="parallel")

    selection_matrix = [0.5, 0.5, 1, 0.5, 0.5, 0.5]
    arm.update_selection_matrix(selection_matrix)

    ee = arm.end_effector()

    p1 = ee.copy()
    p1[2] -= 0.03

    p2 = p1.copy()
    p2[2] += 0.005

    trajectory = p1
    # trajectory = np.stack((p1, p2))
    # trajectory = np.array([-0.02, 0.50, 0.195, -0.00812894,  0.70963372, -0.00882711,  0.70446859])
    target_force = np.zeros(6)

    def f(x): return print(np.round(trajectory[:3] - x[:3], 4))
    arm.zero_ft_sensor()
    res = arm.execute_compliance_control(trajectory, target_wrench=target_force, max_force_torque=[50., 50., 50., 5., 5., 5.],
                                         duration=5, func=f, scale_up_error=True, max_scale_error=1.0)
    print("EE change", ee - arm.end_effector())
    print("ok", np.round(trajectory[:3] - arm.end_effector()[:3], 4))


def move_force():
    """ Linear push. Move until the target force is felt and stop. """
    arm.zero_ft_sensor()

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    target_force = [0, 0, -10, 0, 0, 0]  # express in the end_effector_link

    res = arm.execute_compliance_control(ee, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=15,
                                         stop_on_target_force=True)
    print(res)
    print("EE change", ee - arm.end_effector())


def slicing():
    """ Push down while oscillating in X-axis or Y-axis """
    arm.zero_ft_sensor()

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.03, 0.03, 0.03, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    trajectory = traj_utils.compute_sinusoidal_trajectory(ee, dimension=1, period=3, amplitude=0.02, num_of_points=100)
    target_force = [0, 0, -3, 0, 0, 0]  # express in the end_effector_link

    res = arm.execute_compliance_control(trajectory, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=20)
    print(res)
    print("EE change", ee - arm.end_effector())


def admittance_control():
    """ Spring-mass-damper force control """
    arm.set_control_mode(mode="spring-mass-damper")

    ee = arm.end_effector()
    target_force = np.zeros(6)
    arm.execute_compliance_control(ee, target_wrench=target_force,
                                   max_force_torque=[50., 50., 50., 5., 5., 5.], duration=10,
                                   stop_on_target_force=False)


def free_drive():
    arm.zero_ft_sensor()

    selection_matrix = [0, 0, 0, 0, 0, 0]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.05, 0.05, 0.05, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    target_force = np.zeros(6)

    res = arm.execute_compliance_control(ee, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=15,
                                         stop_on_target_force=False)
    print(res)
    print("EE change", ee - arm.end_effector())


def test():
    # start here
    move_joints()

    for _ in range(3):
        # Move down (cut)
        arm.move_relative(transformation=[0, 0, -0.03, 0, 0, 0], relative_to_tcp=False, duration=0.5, wait=True)

        # Move back up and to the next initial pose
        arm.move_relative(transformation=[0, 0, 0.03, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)
        arm.move_relative(transformation=[0, 0.01, 0, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)


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
    parser.add_argument('-a', '--admittance', action='store_true',
                        help='Spring-mass-damper force control demo')
    parser.add_argument('-s', '--slicing', action='store_true',
                        help='Push down while oscillating on X-axis')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Test')
    parser.add_argument('--namespace', type=str,
                        help='Namespace of arm', default=None)
    args = parser.parse_args()

    rospy.init_node('ur3e_compliance_control')

    ns = "None"
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

    if args.move_cartesian:
        move_cartesian()
    if args.move_force:
        move_force()
    if args.admittance:
        admittance_control()
    if args.free_drive:
        free_drive()
    if args.slicing:
        slicing()
    if args.move_joints:
        move_joints()
    if args.test:
        test()
    # if args.hand_frame_control:
    #     move_hand_frame_control()


if __name__ == "__main__":
    main()
