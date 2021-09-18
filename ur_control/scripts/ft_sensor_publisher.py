#!/usr/bin/env python3

import argparse
import rospy
from ur_control.arm import Arm

from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_RTDE_DRIVER

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    parser.add_argument('--offset', action='store_true', help='offset ft data before start publishing')

    args = parser.parse_args()

    rospy.init_node('ur3_wrench')

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_RTDE_DRIVER

    arm = Arm(ft_sensor=True, driver=driver)
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=args.offset)

    while not rospy.is_shutdown():
        arm.set_wrench_offset(override=False)
        arm.publish_wrench()
        arm.publish_ft_raw()

main()
