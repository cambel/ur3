#!/usr/bin/env python3
# tf2 workaround for Python3
import sys
sys.path[:0] = ['/usr/local/lib/python3.6/dist-packages/'] 

import rospy
import rospkg
import tf
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

from ur_control import transformations
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_RTDE_DRIVER, ROBOT_GAZEBO_DUAL_RIGHT
from ur_control.compliant_controller import CompliantController
import timeit

import sys
import signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


rospy.init_node('ur_handeye_calibration_capture')
listener = tf.TransformListener()
arm = None
tf_data = []

ns = rospy.get_param("ur_calibration_ns", default="/")
ns += "" if ns.endswith("/") else "/"

camera_frame = rospy.get_param( ns + "tracking_base_frame", default="camera_link")
marker_frame = rospy.get_param( ns + "tracking_marker_frame", default="marker_link")
robot_frame = rospy.get_param( ns + "robot_base_frame", default="base_link")
endeffector_frame = rospy.get_param( ns + "robot_effector_frame", default="ee_link")
# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials
packpath = rospack.get_path("ur_handeye_calibration")
savefolder = rospy.get_param( ns + "save_to", default=packpath + '/config/')

def append_tf_data():
    try:
        rospy.sleep(0.1)

        trans, rot = listener.lookupTransform(camera_frame, marker_frame, rospy.Time(0))
        objTocamera = trans + rot
        # print("btc", np.round(objTocamera[:3],4))
        
        trans, rot = listener.lookupTransform(robot_frame, endeffector_frame, rospy.Time(0))
        eeTobase = trans + rot
        # print("bte", np.round(eeTobase[:3],4))

        tf_data.append([objTocamera, eeTobase])
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("Failed =(")

def rotate_wrist(q, changes):
    cq = np.copy(q)
    for change in changes:
        for p in change:
            cq[p[0]] = p[1]
        arm.set_joint_positions(cq, wait=True, t=0.5)
        append_tf_data()
        # input("Enter to continue")


def move_arm(wait=True):
    q = [2.37191, -1.88688, -1.82035,  0.4766 ,  2.31206,  3.18758]
    arm.set_joint_positions(position=q, wait=wait, t=0.5)
    initial_ee = arm.end_effector(q)

    deltas = [
        [0.0, 0.03, 0.08, 0.13, 0.18], # x
        [0.0, 0.03, 0.08, 0.13, 0.18], # y
        [0.0, 0.03, 0.08, 0.13, 0.18], # z
        ]

    X = 5
    Y = 5
    Z = 3

    pose_changes = [
                    [[4, 2.05], [5, 3.20]],
                    [[4, 2.3], [5, 3.20]],
                    [[4, 2.1936], [5, 3.8]],
                    [[4, 2.05], [5, 3.8]],
                    [[4, 2.3], [5, 3.8]],
                    [[4, 2.1936], [5, 2.6]],
                    [[4, 2.05], [5, 2.6]],
                    [[4, 2.3], [5, 2.6]],
                    ]

    for i in range(Z):
        x = y = 0
        dx = 0
        dy = -1
        for _ in range(max(X, Y)**2):
            if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
                delta = np.zeros(6)
                delta[0] = deltas[0][x+1]
                delta[2] = deltas[2][y+1]
                delta[1] = deltas[1][i]
                cpose = transformations.pose_euler_to_quaternion(initial_ee, delta, ee_rotation=False)
                arm.set_target_pose(pose=cpose, wait=True, t=0.5)
                # input("Enter to continue")
                append_tf_data()

                q = arm.joint_angles()
                rotate_wrist(q, pose_changes)

            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                dx, dy = -dy, dx
            x, y = x+dx, y+dy

    np.save(savefolder + "calibration_data_apriltag", tf_data)


def main():
    """ Main function to be run. """
    # driver = ROBOT_GAZEBO_DUAL_RIGHT #ROBOT_GAZEBO
    driver = ROBOT_GAZEBO #ROBOT_GAZEBO

    global arm
    arm = CompliantController(ft_sensor=False, driver=driver, ee_transform=[-0.,   -0.,   0.05,  0,    0.,    0.,    1.  ])

    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    move_arm()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()

