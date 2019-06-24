#!/usr/bin/env python

"""
UR Joint Position Example: keyboard
"""
import argparse

import rospy

from gps.agent.ur.arm import Arm
import ur3_kinematics.arm as ur3_arm
import getch

import numpy as np


def solve_ik(arm, pose):
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
                if (abs(test_ang) <= 2. * np.pi
                        and abs(test_ang - q_guess[i]) <
                        abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        print "ik failed :("
        return None
    best_sol_ind = np.argmin(
        np.sum((weights * (valid_sols - np.array(q_guess)))**2, 1))
    return valid_sols[best_sol_ind]

def map_keyboard(arm):
    arm = arm

    def print_robot_state():
        print("Joint angles:", arm.joint_angles())
        print("End Effector:", arm.end_effector())

    def set_j(joint_name, delta):
        current_position = arm.joint_angles()
        current_position[joint_name] += delta 
        arm.set_joint_positions(current_position, wait=True, t=1.0)
    
    def update_d(delta, increment):
        delta =+ increment

    def set_xyz(dim, delta):
        x = arm.end_effector()
        x[dim] += delta
        q = solve_ik(arm, x)
        arm.set_joint_positions(q, wait=True, t=1.0)

    delta_q = 0.005
    delta_x = 0.005
    
    bindings = {
    #'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    #   key: (function, args, description)
        '4': (set_j, [0, delta_q], "shoulder_pan_joint increase"),
        '1': (set_j, [0, -delta_q], "shoulder_pan_joint decrease"),
        '3': (set_j, [1, delta_q], "shoulder_lift_joint increase"),
        '2': (set_j, [1, -delta_q], "shoulder_lift_joint decrease"),
        'r': (set_j, [2, delta_q], "elbow_joint increase"),
        'q': (set_j, [2, -delta_q], "elbow_joint decrease"),
        'e': (set_j, [3, delta_q], "wrist_1_joint increase"),
        'w': (set_j, [3, -delta_q], "wrist_1_joint decrease"),
        'f': (set_j, [4, delta_q], "wrist_2_joint increase"),
        'a': (set_j, [4, -delta_q], "wrist_2_joint decrease"),
        'd': (set_j, [5, delta_q], "wrist_3_joint increase"),
        's': (set_j, [5, -delta_q], "wrist_3_joint decrease"),
        't': (print_robot_state, [], "right: printing"),
        # Task Space
        'u': (set_xyz, [0, delta_x], "x increase"),
        'i': (set_xyz, [0, -delta_x], "x decrease"),
        'j': (set_xyz, [1, delta_x], "y increase"),
        'k': (set_xyz, [1, -delta_x], "y decrease"),
        'm': (set_xyz, [2, delta_x], "z increase"),
        ',': (set_xyz, [2, -delta_x], "z decrease"),
        # Increase or decrease delta
        '[': (update_d, [delta_x, 0.001], "delta_x increase"),
        ']': (update_d, [delta_x, -0.001], "delta_x decrease"),
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
                print("command: %s" % (cmd[2],))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(bindings.items(),
                                       key=lambda x: x[1][2]):
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
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    args = parser.parse_args()

    rospy.init_node("joint_position_keyboard")
    
    global arm
    if args.robot:
        arm = Arm(ft_sensor=True, real_robot=True)
    else:
        arm = Arm(ft_sensor=True, real_robot=False)
    
    map_keyboard(arm)
    print("Done.")


if __name__ == '__main__':
    main()
