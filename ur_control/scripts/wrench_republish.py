#!/usr/bin/env python3
import argparse
import rospy
import numpy as np
from ur_control.compliant_controller import CompliantController

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument(
        '--right', action='store_true', help='for the dual robot. right arm driver')
    parser.add_argument(
        '--left', action='store_true', help='for the dual robot. left arm driver')
    parser.add_argument('--record', action='store_true', help='record ft data')
    parser.add_argument('--zero', action='store_true', help='record ft data')

    args = parser.parse_args()

    rospy.init_node('ur3e_wrench_republisher')

    ns = ''
    joints_prefix = None
    
    if args.left:
        ns = "left_arm"
        joints_prefix = "leftarm_"
    elif args.right:
        ns = "right_arm"
        joints_prefix = "rightarm_"

    arm = CompliantController(ft_sensor=True, 
                              relative_to_ee=False,
                              namespace=ns,
                              joint_names_prefix=joints_prefix,
                              ft_topic="resense_ft/wrench")
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=args.zero)

    cnt = 0
    offset_cnt = 0
    data = []
    ft_filename = "ft_data.npy"
    while not rospy.is_shutdown():
        arm.publish_wrench()

        if offset_cnt > 100:
            arm.set_wrench_offset(False)
            offset_cnt = 0
        offset_cnt += 1

        if not args.record:
            continue

        if cnt < 500:
            data.append([arm.get_ee_wrench().tolist(), arm.end_effector().tolist()])
            cnt+=1
        else:
            try:
                ldata = np.load(ft_filename, allow_pickle=True)
                ldata = ldata.tolist()
                ldata += data
                np.save(ft_filename, ldata)
                data = []
                cnt = 0
            except IOError:
                np.save(ft_filename, data)

main()