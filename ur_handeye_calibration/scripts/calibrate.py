# tf2 workaround for Python3
import sys
sys.path[:0] = ['/usr/local/lib/python3.6/dist-packages/'] 

import tf
import rospy
import rospkg

from handeye import HandEyeCalibrator, Setup, solver
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
import baldor as br
from pyquaternion import Quaternion
from ur_control import transformations as tr

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials
packpath = rospack.get_path("ur_handeye_calibration")
calibration_file = rospy.get_param( ns + "calibration_file", default=packpath + '/config/calibration_data_apriltag.npy')

def calibrate_simulation(cto_poses, bte_poses):
    calibrator = HandEyeCalibrator(setup=Setup.Fixed)
    for cto_pose, bte_pose in zip(cto_poses, bte_poses):
        bte = tr.pose_to_transform(bte_pose)
        cto = tr.pose_to_transform(cto_pose)
        calibrator.assess_tcp_pose(bte)
        calibrator.add_sample(bte, cto)

    Xest = calibrator.solve(method=solver.Daniilidis1999)
    Xpose = tr.pose_quaternion_from_matrix(Xest)
    print("via Daniilidis1999:", Xpose)

    rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
    assert(rot_rmse > br._FLOAT_EPS)
    assert(trans_rmse > br._FLOAT_EPS)

    rospy.init_node('ur3_force_control')

    tfbr = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        tfbr.sendTransform(Xpose[:3],
                         Xpose[3:],
                         rospy.Time.now(),
                         "camera_es_link",
                         "base_link")
        rate.sleep()



def main():
    """ Main function to be run. """
    
    all_poses = np.load(calibration_file)
    # print(all_poses) 
    cto_poses = all_poses[:,0,:] # camera to object
    bte_poses = all_poses[:,1,:] # base to end-effector

    calibrate_simulation(cto_poses, bte_poses)


if __name__ == "__main__":
    main()

