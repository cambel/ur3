#!/usr/bin/env python
"""
UR Joint Position Example: keyboard
"""
import argparse

import rospy

from ur_control.arm import Arm
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
import ur_control.conversions as conversions

import ur3_kinematics.e_arm as ur3_arm
import getch

import numpy as np
from pyquaternion import Quaternion

import tf
from geometry_msgs.msg import Vector3
np.set_printoptions(suppress=True)


def solve_ik(pose):
    """ Solve IK for ur3 arm 
        pose: [x y z aw ax ay az] array
    """
    pose = np.array(pose).reshape(1, -1)
    current_q = arm.joint_angles()

    ik = ur3_arm.inverse(pose)
    q = best_ik_sol(ik, current_q)

    return current_q if q is None else q


def best_ik_sol(sols, q_guess, weights=np.ones(6)):
    """ Get best IK solution """
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6) * 9999.
        for i in range(6):
            for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * np.pi and abs(test_ang - q_guess[i])
                        < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        print "ik failed :("
        return None
    best_sol_ind = np.argmin(
        np.sum((weights * (valid_sols - np.array(q_guess)))**2, 1))
    return valid_sols[best_sol_ind]


def map_keyboard():
    def print_robot_state():
        print("Joint angles:", np.round(arm.joint_angles(), 5))
        print("End Effector:", np.round(arm.end_effector(rot_type='euler'), 5))
        print("quaternion:", np.round(arm.end_effector()[3:], 6))

    def set_j(joint_name, sign):
        global delta_q
        current_position = arm.joint_angles()
        current_position[joint_name] += delta_q * sign
        arm.set_joint_positions_flex(current_position, t=0.25)

    def update_d(delta, increment):
        if delta == 'q':
            global delta_q
            delta_q += increment
            print "delta_q", delta_q
        if delta == 'x':
            global delta_x
            delta_x += increment
            print "delta_x", delta_x

    def set_pose(dim, sign):
        global delta_x

        J = np.zeros((6, 6))
        twist = np.zeros(6)

        x = arm.end_effector(rot_type='euler')
        qc = arm.joint_angles()

        if dim <= 2:
            twist[dim] += delta_x * sign

        if dim > 2:
            euler = x[3:]
            euler_diff = np.zeros(3)
            euler_diff[dim - 3] += delta_x * sign

            T = conversions.euler_transformation_matrix(euler)
            twist[3:] = np.dot(T, euler_diff)

        J = arm.kinematics.jacobian(arm.joint_angles())
        J_inv = np.linalg.pinv(J)
        dqc = np.dot(J_inv, twist).tolist()[0]

        qc += dqc

        # Publish command
        arm.set_joint_positions_flex(position=qc, t=0.25)

    global delta_q
    global delta_x
    delta_q = 0.0010
    delta_x = 0.0010

    bindings = {
        #'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
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
        'h': (set_pose, [0, 1], "x increase"),
        'k': (set_pose, [0, -1], "x decrease"),
        'y': (set_pose, [1, 1], "y increase"),
        'i': (set_pose, [1, -1], "y decrease"),
        'u': (set_pose, [2, 1], "z increase"),
        'j': (set_pose, [2, -1], "z decrease"),
        'n': (set_pose, [3, 1], "ax increase"),
        'm': (set_pose, [3, -1], "ax decrease"),
        ',': (set_pose, [4, 1], "ay increase"),
        '.': (set_pose, [4, -1], "ay decrease"),
        'o': (set_pose, [5, 1], "az increase"),
        'l': (set_pose, [5, -1], "az decrease"),

        # Increase or decrease delta
        '1': (update_d, ['q', 0.001], "delta_q increase"),
        '2': (update_d, ['q', -0.001], "delta_q decrease"),
        '6': (update_d, ['x', 0.0001], "delta_x increase"),
        '7': (update_d, ['x', -0.0001], "delta_x decrease"),
    }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    while not done and not rospy.is_shutdown():
        c = getch.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
            elif c in bindings:
                cmd = bindings[c]
                #expand binding to something like "set_j(right, 's0', 0.1)"
                cmd[0](*cmd[1])
                print("command: %s" % (cmd[2], ))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(
                        bindings.items(), key=lambda x: x[1][2]):
                    print("  %s: %s" % (key, val[2]))


def main():
    """RSDK Joint Position Example: Keyboard Control

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
        '--robot', action='store_true', help='for the real robot')
    parser.add_argument(
        '--beta', action='store_true', help='for the real robot. beta driver')
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("joint_position_keyboard")

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_MODERN_DRIVER
    elif args.beta:
        driver = ROBOT_UR_RTDE_DRIVER

    global arm
    arm = Arm(ft_sensor=False, driver=driver)

    map_keyboard()
    print("Done.")


if __name__ == '__main__':
    main()
