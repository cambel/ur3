import numpy as np
from pyquaternion import Quaternion

import rospy
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)
from geometry_msgs.msg import (Wrench)

from ur_control import utils, spalg, conversions, transformations
from ur_control.constants import JOINT_ORDER, JOINT_PUBLISHER_REAL, \
    JOINT_PUBLISHER_BETA, JOINT_PUBLISHER_SIM, \
    FT_SUBSCRIBER_REAL, FT_SUBSCRIBER_SIM, \
    ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER, \
    IKFAST, TRAC_IK

from ur_ikfast import ur_kinematics as ur_ikfast

from ur_control.controllers import JointTrajectoryController, FTsensor
from ur_pykdl import ur_kinematics
from trac_ik_python.trac_ik import IK

cprint = utils.TextColors()

class Arm(object):
    """ UR3 arm controller """

    def __init__(self,
                 ft_sensor=False,
                 driver=ROBOT_GAZEBO,
                 ee_transform=None,
                 robot_urdf='ur3e_robot',
                 ik_solver=IKFAST,
                 namespace='ur3'):
        """ ft_sensor bool: whether or not to try to load ft sensor information
            driver string: type of driver to use for real robot or simulation.
                           supported: gazebo, ur_modern_driver, ur_rtde_driver (Newest)
                           use ur_control.constants e.g. ROBOT_GAZEBO
            ee_tranform array [x,y,z,ax,ay,az,w]: optional transformation to the end-effector
                                                  that is applied before doing any operation in task-space
            robot_urdf string: name of the robot urdf file to be used
            namespace string: nodes namespace prefix
        """

        cprint.ok("ft_sensor: {}, driver: {}, ee_transform: {}, \n robot_urdf: {}".format(ft_sensor, driver, ee_transform, robot_urdf))

        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._current_ft = []
        self.ft_sensor = None

        self.ik_solver = ik_solver
        self.ee_transform = ee_transform
        self.ns = namespace

        self.ee_link = 'tool0'
        # self.max_joint_speed = np.deg2rad([120, 120, 120, 220, 220, 220])
        self.max_joint_speed = np.deg2rad([191,191,191,371,371,371])

        self._init_ik_solver(robot_urdf)
        self._init_controllers(driver)
        if ft_sensor:
            self._init_ft_sensor(driver)

### private methods ###

    def _init_controllers(self, driver):
        traj_publisher = None
        if driver == ROBOT_UR_MODERN_DRIVER:
            traj_publisher = JOINT_PUBLISHER_REAL
        elif driver == ROBOT_UR_RTDE_DRIVER:
            traj_publisher = JOINT_PUBLISHER_BETA
        elif driver == ROBOT_GAZEBO:
            traj_publisher = JOINT_PUBLISHER_SIM
        else:
            raise Exception("unsupported driver", driver)
        # Flexible trajectory (point by point)
        traj_publisher_flex = '/' + traj_publisher + '/command'
        cprint.blue("connecting to: {}".format(traj_publisher))
        self._flex_trajectory_pub = rospy.Publisher(traj_publisher_flex,
                                                    JointTrajectory,
                                                    queue_size=10)

        self.joint_traj_controller = JointTrajectoryController(
            publisher_name=traj_publisher)

    def _init_ik_solver(self, robot_urdf):
        self.kdl = ur_kinematics(robot_urdf, ee_link=self.ee_link)

        if self.ik_solver == IKFAST:
            # IKfast libraries
            if robot_urdf == 'ur3_robot':
                self.arm_ikfast = ur_ikfast.URKinematics('ur3')
            elif robot_urdf == 'ur3e_robot':
                self.arm_ikfast = ur_ikfast.URKinematics('ur3e')
        elif self.ik_solver == TRAC_IK:
            self.trac_ik = IK(base_link="base_link", tip_link=self.ee_link,
                              timeout=0.001, epsilon=1e-5, solve_type="Speed",
                              urdf_string=utils.load_urdf_string('ur_pykdl', robot_urdf))
        else:
            raise Exception("unsupported ik_solver", self.ik_solver)

    def _init_ft_sensor(self, driver):
        # Publisher of wrench
        self.pub_ee_wrench = rospy.Publisher('/%s/ee_ft' % self.ns,
                                             Wrench,
                                             queue_size=50)

        if driver == ROBOT_GAZEBO:
            self.ft_sensor = FTsensor(namespace=FT_SUBSCRIBER_SIM)
        else:
            self.ft_sensor = FTsensor(namespace=FT_SUBSCRIBER_REAL)
        rospy.sleep(1)
        self.set_wrench_offset(override=False)


    def _update_wrench_offset(self):
        self.wrench_offset = self.get_filtered_ft().tolist()
        rospy.set_param('/%s/ft_offset' % self.ns, self.wrench_offset)

    def _flexible_trajectory(self, position, time=5.0, vel=None):
        """ Publish point by point making it more flexible for real-time control """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = JOINT_ORDER

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        target.positions = position

        # These times determine the speed at which the robot moves:
        if vel is not None:
            target.velocities = [vel] * 6

        target.time_from_start = rospy.Duration(time)

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        self._flex_trajectory_pub.publish(action_msg)

    def _solve_ik(self, pose):
        if self.ee_transform is not None:
            inv_ee_transform = np.copy(self.ee_transform)
            inv_ee_transform[:3] *= -1
            pose = np.array(conversions.transform_end_effector(pose, inv_ee_transform))

        if self.ik_solver == IKFAST:
            ik = self.arm_ikfast.inverse(pose, q_guess=self.joint_angles())

        elif self.ik_solver == TRAC_IK:
            ik = self.ik_solver.get_ik(self.joint_angles(), pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])
            if ik is None:
                print("IK not found")

        return ik

### Public methods ###

    def get_filtered_ft(self):
        """ Get measurements from FT Sensor.
            Measurements are filtered with a low-pass filter.
            Measurements are given in sensors orientation.
        """
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")

        ft_limitter = [300, 300, 300, 30, 30, 30]  # Enforce measurement limits (simulation)
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
            self.wrench_offset = rospy.get_param('/%s/ft_offset' % self.ns, None)
            if self.wrench_offset is None:
                self._update_wrench_offset()

    def get_ee_wrench(self):
        """ Compute the wrench (force/torque) in task-space """
        if self.ft_sensor is None:
            return np.zeros(6)

        wrench_force = self.ft_sensor.get_filtered_wrench()

        # compute force transformation?
        if self.wrench_offset is not None:
            wrench_force = np.array(wrench_force) - np.array(
                self.wrench_offset)

        # # # Transform of EE
        pose = self.end_effector()
        ee_transform = Quaternion(np.roll(pose[3:], 1)).transformation_matrix

        # # # Wrench force transformation
        wFtS = spalg.force_frame_transform(ee_transform)
        wrench = np.dot(wFtS, wrench_force)

        return wrench

    def publish_wrench(self):
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")

        " Publish arm's end-effector wrench "
        wrench = self.get_ee_wrench()
        # Note you need to call rospy.init_node() before this will work
        self.pub_ee_wrench.publish(conversions.to_wrench(wrench))

    def end_effector(self,
                     joint_angles=None,
                     rot_type='quaternion'):
        """ Return End Effector Pose """

        joint_angles = self.joint_angles() if joint_angles is None else joint_angles

        if rot_type == 'quaternion':
            # forward kinematics
            if self.ik_solver == IKFAST:
                x = self.arm_ikfast.forward(joint_angles)
            else:
                x = self.kdl.forward_position_kinematics(joint_angles)

            # apply extra transformation of end-effector
            if self.ee_transform is not None:
                x = np.array(conversions.transform_end_effector(x, self.ee_transform))
            return x

        elif rot_type == 'euler':
            x = self.end_effector(joint_angles)
            euler = np.array(transformations.euler_from_quaternion(x[3:], axes='rxyz'))
            return np.concatenate((x[:3], euler))

        else:
            raise Exception("Rotation Type not supported", rot_type)

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

### Basic Control Methods ###

    def set_joint_positions(self,
                            position,
                            velocities=None,
                            accelerations=None,
                            wait=False,
                            t=5.0):
        self.joint_traj_controller.add_point(positions=position,
                                             time=t,
                                             velocities=velocities,
                                             accelerations=accelerations)
        self.joint_traj_controller.start(delay=1., wait=wait)
        self.joint_traj_controller.clear_points()

    def set_joint_positions_flex(self, position, t=5.0, v=None):
        qc = self.joint_angles()
        deltaq = (qc - position)
        speed = deltaq / t
        cmd = position
        if np.any(np.abs(speed) > (self.max_joint_speed/t)):
            print("exceeded max speed", speed)
            return
        self._flexible_trajectory(cmd, t, v)

    def set_target_pose(self, pose, wait=False, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self._solve_ik(pose)
        if q is None:
            # IK not found
            return False
        else:
            self.set_joint_positions(q, wait=wait, t=t)
            return True

    def set_target_pose_flex(self, pose, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self._solve_ik(pose)
        if q is None:
            # IK not found
            return False
        else:
            self.set_joint_positions_flex(q, t=t)
            return True
