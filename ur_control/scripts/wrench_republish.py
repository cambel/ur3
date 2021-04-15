#!/usr/bin/env python
import argparse
import rospy
import numpy as np
from ur_control.compliant_controller import CompliantController

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm', default=None)
    parser.add_argument('--record', action='store_true', help='record ft data')
    parser.add_argument('--zero', action='store_true', help='record ft data')

    args = parser.parse_args()

    rospy.init_node('ur3e_wrench_republisher')

    ns = ''
    joints_prefix = None
    robot_urdf = "ur3e_robot"
    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
    
    extra_ee = [0, 0, 0.0, 0, 0, 0, 1]

    global arm
    arm = CompliantController(ft_sensor=True, ee_transform=extra_ee, 
              namespace=ns, 
              joint_names_prefix=joints_prefix, 
              robot_urdf=robot_urdf)
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=args.zero)

    offset_cnt = 0

    while not rospy.is_shutdown():
        arm.publish_wrench()

        # if offset_cnt > 100:
        #     arm.set_wrench_offset(False)
        #     offset_cnt = 0
        # offset_cnt += 1


main()
