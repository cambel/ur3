import numpy as np
from pyquaternion import Quaternion

import rospy
import utils as utils
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

from ur_control.constants import JOINT_ORDER, JOINT_PUBLISHER_REAL, \
                                 JOINT_PUBLISHER_BETA, JOINT_PUBLISHER_SIM, \
                                 FT_SUBSCRIBER_REAL, FT_SUBSCRIBER_SIM, \
                                 ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER

import ur3_kinematics.arm as ur3_arm
import ur3_kinematics.e_arm as ur3e_arm

from controllers import JointTrajectoryController, FTsensor
import spalg as spalg
from ur_pykdl import ur_kinematics

class Arm(object):
    """ UR3 arm controller """

    def __init__(self, ft_sensor=False, robot="simulation", ee_transform=[0, 0, 0, 0, 0, 0, 1], robot_urdf='ur3e_robot'):
        """ Constructor 
        
            ft_sensor bool: whether or not to try to load ft sensor information

            real_robot bool: where or not to use the node ids for the real robot

            ee_tranform array [x,y,z,ax,ay,az,w]: optional transformation to the end-effector 
                    that is applied before doing any operation in task space

            robot_urdf string: name of the robot urdf file to be used

        """
        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._current_ft = []

        self.kinematics = ur_kinematics(robot_urdf)

        # IKfast libraries
        if robot_urdf == 'ur3_robot':
            self.arm_ikfast = ur3_arm
        elif robot_urdf == 'ur3e_robot':
            self.arm_ikfast = ur3e_arm

        self.ft_sensor = None
    
        # Publisher of wrench
        self.pub_ee_wrench = rospy.Publisher(
            '/ur3/ee_ft', WrenchStamped, queue_size=10)

        # We need the special end effector link for adjusting the wrench directions
        self.ee_transform = ee_transform

        traj_publisher = None
        if robot == ROBOT_UR_MODERN_DRIVER:
            traj_publisher = JOINT_PUBLISHER_REAL
        elif robot == ROBOT_UR_RTDE_DRIVER:
            traj_publisher = JOINT_PUBLISHER_BETA
        elif robot == ROBOT_GAZEBO:
            traj_publisher = JOINT_PUBLISHER_SIM
        else:
            raise Exception("invalid driver")

        traj_publisher_flex = '/' + traj_publisher + '/command'

        print "connecting to", traj_publisher
        # Flexible trajectory (point by point)
        self._flex_trajectory_pub = rospy.Publisher(traj_publisher_flex, JointTrajectory, queue_size=10)
        self.joint_traj_controller = JointTrajectoryController(publisher_name=traj_publisher)

        # FT sensor data
        if ft_sensor:
            if robot == ROBOT_GAZEBO:
                self.ft_sensor = FTsensor(namespace=FT_SUBSCRIBER_SIM)
            else:
                self.ft_sensor = FTsensor(namespace=FT_SUBSCRIBER_REAL)
            self.wrench_offset = None
            rospy.sleep(1)

        
    def _flexible_trajectory(self, position, time=5.0, vel=None):
        """ Publish point by point making it more flexible for real-time control """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = JOINT_ORDER

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        target.positions = position

        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        if vel is not None:
            target.velocities = [vel] * 6

        target.time_from_start = rospy.Duration(time)

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        self._flex_trajectory_pub.publish(action_msg)

    def get_ft_measurements(self):
        " Get measurements from FT Sensor "
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")
            
        ft_limitter = [100, 100, 100, 30, 30, 30]
        ft = self.ft_sensor.get_filtered_wrench()
        ft = [
            ft[i] if abs(ft[i]) < ft_limitter[i] else ft_limitter[i]
            for i in range(6)
        ]
        return np.array(ft)

    def set_wrench_offset(self, override=False):
        """ Set a wrench offset """
        if override:
            self._update_wrench_offset()
        else:    
            self.wrench_offset = rospy.get_param('/ur3/ft_offset', None)
            if self.wrench_offset is None:
                self._update_wrench_offset()
                
    def _update_wrench_offset(self):
        self.wrench_offset = self.get_ft_measurements().tolist()
        rospy.set_param('/ur3/ft_offset', self.wrench_offset)

    def get_ee_wrench(self):
        """ Get the wrench (force/torque) in task-space """
        wrench_force = self.ft_sensor.get_filtered_wrench()

        if self.wrench_offset is not None:
            wrench_force = np.array(wrench_force) - np.array(self.wrench_offset)

        q_actual = self.joint_angles()

        # Transform of EE
        ee_transform = self.kinematics.end_effector_transform(q_actual)
        
        # Wrench force transformation
        wFtS = spalg.force_frame_transform(ee_transform)
        
        return np.dot(wFtS, wrench_force)

    def publish_wrench(self):
        wrench = self.get_ee_wrench()
        msg = WrenchStamped()
        # Note you need to call rospy.init_node() before this will work
        msg.header.stamp = rospy.Time.now()
        msg.wrench.force.x = wrench[0]
        msg.wrench.force.y = wrench[1]
        msg.wrench.force.z = wrench[2]
        msg.wrench.torque.x = wrench[3]
        msg.wrench.torque.x = wrench[4]
        msg.wrench.torque.z = wrench[5]
        self.pub_ee_wrench.publish(msg)

    def end_effector(self):
        return self.kinematics.forward_position_kinematics(self.joint_angles())

    def joint_angle(self, joint):
        """
        Return the requested joint angle.

        @type joint: str
        @param joint: name of a joint
        @rtype: float
        @return: angle in radians of individual joint
        """
        return self.joint_traj_controller.get_joint_positions()[joint]

    def joint_angles(self):
        """
        Return all joint angles.

        @rtype: dict({str:float})
        @return: unordered dict of joint name Keys to angle (rad) Values
        """
        return self.joint_traj_controller.get_joint_positions()

    def joint_velocity(self, joint):
        """
        Return the requested joint velocity.

        @type joint: str
        @param joint: name of a joint
        @rtype: float
        @return: velocity in radians/s of individual joint
        """
        return self.joint_traj_controller.get_joint_velocities()[joint]

    def joint_velocities(self):
        """
        Return all joint velocities.

        @rtype: dict({str:float})
        @return: unordered dict of joint name Keys to velocity (rad/s) Values
        """
        return self.joint_traj_controller.get_joint_velocities()

    def set_joint_positions(self,
                            position,
                            velocities=None,
                            accelerations=None,
                            wait=False,
                            t=5.0):
        self.joint_traj_controller.add_point(
            positions=position,
            time=t,
            velocities=velocities,
            accelerations=accelerations)
        self.joint_traj_controller.start(delay=1., wait=wait)
        self.joint_traj_controller.clear_points()

    def set_joint_positions_flex(self, position, t=5.0, v=None):
        self._flexible_trajectory(position, t, v)

    def set_target_pose(self, pose, wait=False, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self.solve_ik(pose)
        self.set_joint_positions(q, wait=wait, t=t)

    def set_target_pose_flex(self, pose, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self.solve_ik(pose)

        self.set_joint_positions_flex(q, t=t)

    def solve_ik(self, pose):
        """ Solve IK for ur3 arm 
            pose: [x y z aw ax ay az] array
        """
        pose = self.transform_end_effector(pose, self.ee_transform)
        pose = np.array(pose).reshape(1, -1)
        current_q = self.joint_angles()
        
        ik = self.arm_ikfast.inverse(pose)
        q = self._best_ik_sol(ik, current_q)
        
        return self.joint_angles() if q is None else q

    def _best_ik_sol(self, sols, q_guess, weights=np.ones(6)):
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

    def transform_end_effector(self, pose, extra_pose, matrix=False):
        """ 
        Transform end effector pose
         pose: current pose [x, y, z, ax, ay, az, w]
         extra_pose: additional transformation [x, y, z, ax, ay, az, w]
         matrix: if true: return (translation, rotation matrix)
                 else: return translation + quaternion list
        """
        extra_translation = np.array(extra_pose[:3]).reshape(3, 1)
        extra_rot = Quaternion(np.roll(extra_pose[3:], 1)).rotation_matrix

        c_trans = np.array(pose[:3]).reshape(3, 1)
        c_rot = Quaternion(np.roll(
            pose[3:],
            1)).rotation_matrix  # BE CAREFUL!! Pose from KDL is ax ay az aw
        #              Pose from IKfast is aw ax ay az

        n_trans = np.matmul(c_rot, extra_translation) + c_trans
        n_rot = np.matmul(c_rot, extra_rot)

        if matrix:
            return n_trans.flatten(), n_rot

        return n_trans.flatten().tolist() + np.roll(
            Quaternion(matrix=n_rot).normalised.elements, -1).tolist()
