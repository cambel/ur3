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
UR Joint Position Example: 3Dconnexion mouse

requires ros-$ROS-VERSION-spacenav-node
and to launch roslaunch spacenav_node classic.launch
#TODO launch automatically
"""
import argparse

import rospy

from ur_control.arm import Arm
from ur_control.mouse_6d import Mouse6D
from ur_control import transformations

from ur_ikfast import ur_kinematics

import numpy as np

np.set_printoptions(suppress=True)
ur3e_arm = ur_kinematics.URKinematics('ur3e')
mouse6d = Mouse6D()

axes = 'rxyz'


def e2q(e):
    return transformations.quaternion_from_euler(e[0], e[1], e[2], axes=axes)


def print_robot_state():
    print(("Joint angles:", np.round(arm.joint_angles(), 3)))
    print(("End Effector:", np.round(arm.end_effector(rot_type='euler'), 3)))


def start_control(motion_type="linear"):
    print("Start moving. type", motion_type)
    rate = rospy.Rate(125)
    delta_x = 0.01
    delta_q = np.deg2rad(1)
    while not rospy.is_shutdown():
        x = arm.end_effector()
        xd = np.array(mouse6d.twist)

        xd[:3] = [delta_x*np.sign(xd[i]) if abs(xd[i]) > 0.15 else 0.0 for i in range(3)]
        xd[3:] = [delta_q*np.sign(xd[3+i]) if abs(xd[3+i]) > 0.15 else 0.0 for i in range(3)]
        if motion_type == "rotated":
            xd[2] *= -1
        elif motion_type == "linear":
            pass
        else:
            print("motion_type not supported", motion_type)
            break

        x = transformations.pose_from_angular_velocity(x, xd, dt=0.25)
        if mouse6d.joy_buttons[0] == 1:
            print_robot_state()

        arm.set_target_pose_flex(pose=x, t=0.25)
        rate.sleep()


def main():
    """ 3D mouse Control """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument('-r', action='store_true', help='move using relative rotation of end-effector')
    parser.add_argument(
        '--robot', action='store_true', help='for the real robot')
    parser.add_argument(
        '--beta', action='store_true', help='for the real robot. beta driver')
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("joint_position_keyboard")

    global arm
    arm = Arm(ft_sensor=False)

    start_control()
    print("Done.")


if __name__ == '__main__':
    main()
