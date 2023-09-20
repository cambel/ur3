#! /usr/bin/env python

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
 
from ur_control import transformations, traj_utils, conversions
from ur_control.arm import Arm
from ur_control.constants import GENERIC_GRIPPER
import argparse
import random
import rospy
import timeit

# If Docker version of Python3 version is used in melodic, 
# Install tf for python3 like this: pip install --extra-index-url https://rospypi.github.io/simple/ tf2_ros
# Then enable the following  lines to re-direct tf to the new library
import sys
sys.path[:0] = ['/usr/local/lib/python3.6/dist-packages/']
import tf

import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def move_joints(wait=True):
    # desired joint configuration 'q'
    q = [0, 0, 0, 0, 0, 0]
    q = [3.2317, -1.979, 1.3969, -0.4844, -0.1151, -1.7565]
    q = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]

    # go to desired joint configuration
    # in t time (seconds)
    # wait is for waiting to finish the motion before executing
    # anything else or ignore and continue with whatever is next
    arm.set_joint_positions(position=q, wait=wait, t=0.5)


def follow_trajectory():
    traj = [
        [2.4463, -1.8762, -1.6757, 0.3268, 2.2378, 3.1960],
        [2.5501, -1.9786, -1.5293, 0.2887, 2.1344, 3.2062],
        [2.5501, -1.9262, -1.3617, 0.0687, 2.1344, 3.2062],
        [2.4463, -1.8162, -1.5093, 0.1004, 2.2378, 3.1960],
        [2.3168, -1.7349, -1.6096, 0.1090, 2.3669, 3.1805],
        [2.3168, -1.7997, -1.7772, 0.3415, 2.3669, 3.1805],
        [2.3168, -1.9113, -1.8998, 0.5756, 2.3669, 3.1805],
        [2.4463, -1.9799, -1.7954, 0.5502, 2.2378, 3.1960],
        [2.5501, -2.0719, -1.6474, 0.5000, 2.1344, 3.2062],
    ]
    for t in traj:
        arm.set_joint_positions(position=t, wait=True, t=1.0)


def move_endeffector(wait=True):
    # get current position of the end effector
    cpose = arm.end_effector()
    # define the desired translation/rotation
    deltax = np.array([0., 0., 0.04, 0., 0., 0.])
    # add translation/rotation to current position
    cpose = transformations.pose_euler_to_quaternion(cpose, deltax, ee_rotation=True)
    # execute desired new pose
    # may fail if IK solution is not found
    arm.set_target_pose(pose=cpose, wait=True, t=1.0)


def move_gripper():
    print("closing")
    arm.gripper.close()
    rospy.sleep(1.0)
    print("opening")
    arm.gripper.open()
    rospy.sleep(1.0)
    print("moving")
    arm.gripper.command(0.5, percentage=True)  # in percentage (80%)
    # 0.0 is full close, 1.0 is full open
    rospy.sleep(1.0)
    print("moving")
    arm.gripper.command(0.01)  # in meters
    # 0.05 is full open, 0.0 is full close
    # max gap for the Robotiq Hand-e is 0.05 meters

    print("current gripper position", round(arm.gripper.get_position(), 4), "meters")


def grasp_naive():
    # probably won't work
    arm.gripper.open()
    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)

    q2 = [1.82225, -1.55525,  1.86741, -2.03039, -1.60938,  0.24935]
    arm.set_joint_positions(q2, wait=True, t=1.0)

    arm.gripper.command(0.036)
    rospy.sleep(0.5)

    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)


def grasp_plugin():
    arm.gripper.open()
    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)

    q2 = [1.82225, -1.55525,  1.86741, -2.03039, -1.60938,  0.24935]
    arm.set_joint_positions(q2, wait=True, t=1.0)

    arm.gripper.command(0.039)
    # attach the object "link" to the robot "model_name"::"link_name"
    arm.gripper.grab(link_name="cube3::link")

    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)
    rospy.sleep(2.0)  # release after 2 secs

    # dettach the object "link" to the robot "model_name"::"link_name"
    arm.gripper.open()
    arm.gripper.release(link_name="cube3::link")


def get_random_valid_direction(plane):
    if plane == "XZ":
        return random.choice(["+X", "-X", "+Z", "-Z"])
    elif plane == "YZ":
        return random.choice(["+Y", "-Y", "+Z", "-Z"])
    elif plane == "XY":
        return random.choice(["+X", "-X", "+Y", "-Y"])
    else:
        raise ValueError("Invalid value for plane: %s" % plane)

def circular_trajectory():
    """ Simple circular trajectory from initial pose. 5cm of radius"""
    initial_q = [1.8391, -1.5659, 1.4889, -1.6421, -1.6115, 0.2656]
    arm.set_joint_positions(initial_q, wait=True, t=2)

    duration = 5.0
    steps = 100
    plane = "XY"
    direction = get_random_valid_direction(plane)
    dummy_trajectory = traj_utils.compute_trajectory(initial_pose=[0, 0, 0, 0, 0, 0, 1.],
                                                    plane=plane, radius=0.05, 
                                                    radius_direction=direction, steps=steps, revolutions=1,
                                                    from_center=False, trajectory_type="circular")



    listener = tf.TransformListener()
    # convert dummy_trajectory (initial pose frame id) to robot's base frame
    try:
        listener.waitForTransform("base_link", "wrist_3_link", rospy.Time(0), rospy.Duration(1))
        transform2target = listener.fromTranslationRotation(*listener.lookupTransform("base_link", "wrist_3_link", rospy.Time(0)))
    except Exception as e:
        print(e)
        return False

    for p in dummy_trajectory:
        ps = conversions.to_pose_stamped("base_link", p)
        next_pose = conversions.from_pose_to_list(conversions.transform_pose("base_link", transform2target, ps).pose)
        print("next_pose", np.round(next_pose[:3].tolist(),4))
        
        arm.set_target_pose_flex(pose=next_pose, t=duration/steps)
        rospy.sleep(duration/steps)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-t', '--move_traj', action='store_true',
                        help='move following a trajectory of joint configurations')
    parser.add_argument('-e', '--move_ee', action='store_true',
                        help='move to a desired end-effector position')
    parser.add_argument('-g', '--gripper', action='store_true',
                        help='Move gripper')
    parser.add_argument('--grasp_naive', action='store_true',
                        help='Test simple grasping (cube_tasks world)')
    parser.add_argument('--grasp_plugin', action='store_true',
                        help='Test grasping plugin (cube_tasks world)')
    parser.add_argument('--circle', action='store_true',
                        help='Circular rotation around a target pose')

    args = parser.parse_args()

    rospy.init_node('ur3e_script_control')

    global arm
    arm = Arm(
        ft_sensor=True,  # get Force/Torque data or not
        gripper=GENERIC_GRIPPER,  # Enable gripper
        )

    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    if args.move:
        move_joints()
    if args.move_traj:
        follow_trajectory()
    if args.move_ee:
        move_endeffector()
    if args.gripper:
        move_gripper()
    if args.grasp_naive:
        grasp_naive()
    if args.grasp_plugin:
        grasp_plugin()
    if args.circle:
        circular_trajectory()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
