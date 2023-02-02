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

"""
UR Joint Position Example: keyboard
"""
import argparse

import rospy

from ur_control.arm import Arm
from ur_control import transformations

import getch

import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


def map_keyboard():
    def print_robot_state():
        print("Joint angles:", np.round(arm.joint_angles(), 4).tolist())
        print("EE Pose:", np.round(arm.end_effector(), 5).tolist())
        if arm.gripper:
            print("Gripper position:", np.round(arm.gripper.get_position(), 4))

    def set_j(joint_name, sign):
        global delta_q
        current_position = arm.joint_angles()
        current_position[joint_name] += delta_q * sign
        arm.set_joint_positions_flex(current_position, t=0.25)

    def update_d(delta, increment):
        if delta == 'q':
            global delta_q
            delta_q += np.deg2rad(increment)
            print(("delta_q", np.rad2deg(delta_q)))
        if delta == 'x':
            global delta_x
            delta_x += increment
            print(("delta_x", delta_x))

    def set_pose_ik(dim, sign):
        global delta_x
        global delta_q

        x = arm.end_effector()
        delta = np.zeros(6)

        n = 500
        dt = 0.25/float(n)

        if dim <= 2:  # position
            delta[dim] += delta_x * sign / 0.25
        else:  # rotation
            delta[dim] += delta_q * sign / 0.25

        for _ in range(n):
            x = transformations.pose_from_angular_velocity(x, delta, dt=dt, rotated_frame=relative_ee)

        arm.set_target_pose_flex(pose=x, t=0.25)

    def open_gripper():
        arm.gripper.open()

    def close_gripper():
        arm.gripper.close()

    def move_gripper(delta):
        cpose = arm.gripper.get_position()
        cpose += delta
        arm.gripper.command(cpose)

    global delta_q
    global delta_x
    delta_q = np.deg2rad(1.0)
    delta_x = 0.005

    bindings = {
        #   key: (function, args, description)
        'z': (set_j, [0, 1], "shoulder_pan_joint increase"),
        'v': (set_j, [0, -1], "shoulder_pan_joint decrease"),
        'x': (set_j, [1, 1], "shoulder_lift_joint increase"),
        'c': (set_j, [1, -1], "shoulder_lift_joint decrease"),
        'a': (set_j, [2, 1], "elbow_joint increase"),
        'f': (set_j, [2, -1], "elbow_joint decrease"),
        's': (set_j, [3, 1], "wrist_1_joint increase"),
        'd': (set_j, [3, -1], "wrist_1_joint decrease"),
        'q': (set_j, [4, 1], "wrist_2_joint increase"),
        'r': (set_j, [4, -1], "wrist_2_joint decrease"),
        'w': (set_j, [5, 1], "wrist_3_joint increase"),
        'e': (set_j, [5, -1], "wrist_3_joint decrease"),
        'p': (print_robot_state, [], "right: printing"),
        # Task Space
        'h': (set_pose_ik, [0, 1], "x increase"),
        'k': (set_pose_ik, [0, -1], "x decrease"),
        'y': (set_pose_ik, [1, 1], "y increase"),
        'i': (set_pose_ik, [1, -1], "y decrease"),
        'u': (set_pose_ik, [2, 1], "z increase"),
        'j': (set_pose_ik, [2, -1], "z decrease"),
        'n': (set_pose_ik, [3, 1], "ax increase"),
        'm': (set_pose_ik, [3, -1], "ax decrease"),
        ',': (set_pose_ik, [4, 1], "ay increase"),
        '.': (set_pose_ik, [4, -1], "ay decrease"),
        'o': (set_pose_ik, [5, 1], "az increase"),
        'l': (set_pose_ik, [5, -1], "az decrease"),

        # Increase or decrease delta
        '1': (update_d, ['q', 0.25], "delta_q increase"),
        '2': (update_d, ['q', -0.25], "delta_q decrease"),
        '6': (update_d, ['x', 0.0001], "delta_x increase"),
        '7': (update_d, ['x', -0.0001], "delta_x decrease"),

        # Gripper
        '5': (move_gripper, [0.002], "open gripper a bit"),
        't': (open_gripper, [], "open gripper"),
        'g': (close_gripper, [], "close gripper"),
        'b': (move_gripper, [-0.002], "close gripper a bit"),
    }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    while not done and not rospy.is_shutdown():
        c = getch.getch()
        if c:
            # catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
            elif c in bindings:
                cmd = bindings[c]
                # expand binding to something like "set_j(right, 's0', 0.1)"
                cmd[0](*cmd[1])
                print(("command: %s" % (cmd[2], )))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(
                        list(bindings.items()), key=lambda x: x[1][2]):
                    print(("  %s: %s" % (key, val[2])))


def main():
    """Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on one of Baxter's arms. Each arm is represented
    by one side of the keyboard and inner/outer key pairings
    on each row for each joint.
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__, epilog=epilog)
    parser.add_argument(
        '--relative', action='store_true', help='Motion Relative to ee')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm (useful when having multiple arms)', default=None)
    parser.add_argument(
        '--gripper', action='store_true', help='enable gripper commands')
    parser.add_argument(
        '--robot', type=str, help='Version of Universal Robot arm. Default="ur3e"', default='ur3e')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("joint_position_keyboard", log_level=rospy.INFO)

    global relative_ee
    relative_ee = args.relative

    ns = args.namespace
    robot_urdf = "ur3e"
    rospackage = None
    tcp_link = "gripper_tip_link"
    use_gripper = args.gripper
    joint_names_prefix = ns+'_' if ns else ''

    global arm
    arm = Arm(gripper=use_gripper, namespace=ns,
              joint_names_prefix=joint_names_prefix,
              robot_urdf=robot_urdf, robot_urdf_package=rospackage,
              ee_link=tcp_link,
              ft_topic='wrench')

    map_keyboard()
    print("Done.")


if __name__ == '__main__':
    main()
