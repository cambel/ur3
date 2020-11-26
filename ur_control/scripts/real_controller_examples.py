#!/usr/bin/env python
from ur_control import utils, spalg, transformations
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
from ur_control.impedance_control import AdmittanceModel
from ur_control.compliant_controller import CompliantController
import argparse
import rospy
import timeit
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def move_joints(wait=True):
    # desired joint configuration 'q'
    q = [2.37191, -1.88688, -1.82035,  0.4766,  2.31206,  3.18758]

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
    # very different than simulation
    from robotiq_urcap_control.msg import Robotiq2FGripper_robot_input as inputMsg
    from robotiq_urcap_control.msg import Robotiq2FGripper_robot_output as outputMsg
    from robotiq_urcap_control.robotiq_urcap_control import RobotiqGripper
    print("Connecting to gripper")
    robot_ip = rospy.get_param("/ur_hardware_interface/robot_ip")
    gripper = RobotiqGripper(robot_ip=robot_ip)
    # The Gripper status is published on the topic named 'Robotiq2FGripperRobotInput'
    pub = rospy.Publisher('Robotiq2FGripperRobotInput', inputMsg, queue_size=1)

    # The Gripper command is received from the topic named 'Robotiq2FGripperRobotOutput'
    rospy.Subscriber('Robotiq2FGripperRobotOutput', outputMsg, gripper.send_command)

    gripper.connect()
    gripper.activate()
    # gripper.move_and_wait_for_pos(position=100, speed=10, force=10)
    # gripper.move_and_wait_for_pos(position=0, speed=10, force=10)
    # gripper.move_and_wait_for_pos(position=255, speed=10, force=10)
    gripper.disconnect()


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
    parser.add_argument('-r', '--rotation', action='store_true',
                        help='Rotation slerp')
    parser.add_argument('--relative', action='store_true', help='relative to end-effector')
    parser.add_argument('--rotation_pd', action='store_true', help='relative to end-effector')

    args = parser.parse_args()

    rospy.init_node('ur3e_script_control')

    global arm
    arm = CompliantController(
        ft_sensor=True,  # get Force/Torque data or not
        driver=ROBOT_UR_RTDE_DRIVER,  # which controller (sim?, robot?)
        ee_transform=[-0., -0., 0.21, 0, 0., 0., 1.],  # transformation for the tip of the robot
        gripper=True,  # Enable gripper
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

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
