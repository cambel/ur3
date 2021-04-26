#!/usr/bin/env python
import argparse
import rospy
import numpy as np
from ur_control.compliant_controller import CompliantController
from ur_control import transformations
def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm', default=None)
    parser.add_argument('--zero', action='store_true', help='reset ft at start')
    parser.add_argument('--relative', action='store_true', help='FT relative to EE')

    args = parser.parse_args()

    rospy.init_node('ur3e_wrench_republisher')

    ns = ''
    joints_prefix = None
    robot_urdf = "ur3e_robot"
    rospackage = None

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
        rospackage = "o2ac_scene_description"
    
    extra_ee = [0,0,0.] + transformations.quaternion_from_euler(*[np.pi/4,0,0]).tolist()
    extra_ee = [0.0, 0.0, 0.173, 0.500, -0.500, 0.500, 0.500]

    global arm
    arm = CompliantController(ft_sensor=True, ee_transform=extra_ee, 
              namespace=ns, 
              joint_names_prefix=joints_prefix, 
              robot_urdf=robot_urdf, robot_urdf_package=rospackage, relative_to_ee=args.relative)
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=args.zero)

    offset_cnt = 0

    while not rospy.is_shutdown():
        arm.publish_wrench(relative=args.relative)

        if offset_cnt > 100:
            arm.set_wrench_offset(False)
            offset_cnt = 0
        offset_cnt += 1


main()
