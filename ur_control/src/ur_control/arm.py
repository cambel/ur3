import numpy as np
from pyquaternion import Quaternion

import rospy
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)
from geometry_msgs.msg import (Wrench)

from ur_control import utils, spalg, conversions, transformations
from ur_control.constants import JOINT_ORDER, JOINT_PUBLISHER_ROBOT, FT_SUBSCRIBER, IKFAST, TRAC_IK, \
    DONE, FORCE_TORQUE_EXCEEDED, SPEED_LIMIT_EXCEEDED, IK_NOT_FOUND, get_arm_joint_names, \
    BASE_LINK, EE_LINK

try:
    from ur_ikfast import ur_kinematics as ur_ikfast
except ImportError:
    print("Import ur_ikfast not available, IKFAST would not be supported without it")

from ur_control.controllers import JointTrajectoryController, FTsensor, GripperController
from ur_pykdl import ur_kinematics
from trac_ik_python.trac_ik import IK

cprint = utils.TextColors()


class Arm(object):
    """ UR3 arm controller """

    def __init__(self,
                 ft_sensor=False,
                 ee_transform=None,
                 robot_urdf='ur3e_robot',
                 robot_urdf_package=None,
                 ik_solver=TRAC_IK,
                 namespace='',
                 gripper=False,
                 joint_names_prefix=None,
                 ft_topic=None,
                 base_link=None,
                 ee_link=None):
        """ ft_sensor bool: whether or not to try to load ft sensor information
            ee_tranform array [x,y,z,ax,ay,az,w]: optional transformation to the end-effector
                                                  that is applied before doing any operation in task-space
            robot_urdf string: name of the robot urdf file to be used
            namespace string: nodes namespace prefix
            gripper bool: enable gripper control
        """

        cprint.ok("ft_sensor: {}, ee_transform: {}, \n robot_urdf: {}".format(ft_sensor, ee_transform, robot_urdf))

        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._current_ft = []

        self._robot_urdf = robot_urdf
        self._robot_urdf_package = robot_urdf_package if robot_urdf_package is not None else 'ur_pykdl'

        self.ft_sensor = None
        self.ft_topic = ft_topic if ft_topic is not None else FT_SUBSCRIBER

        self.ik_solver = ik_solver
        self.ee_transform = ee_transform
        assert namespace is not None, "namespace cannot be None"
        self.ns = namespace
        self.joint_names_prefix = joint_names_prefix

        _base_link = base_link if base_link is not None else BASE_LINK
        _ee_link = ee_link if ee_link is not None else EE_LINK

        self.base_link = BASE_LINK if joint_names_prefix is None else joint_names_prefix + BASE_LINK
        self.ee_link = EE_LINK if joint_names_prefix is None else joint_names_prefix + EE_LINK
        
        self.max_joint_speed = np.deg2rad([100, 100, 100, 200, 200, 200])
        # self.max_joint_speed = np.deg2rad([191, 191, 191, 371, 371, 371])

        self._init_ik_solver()
        self._init_controllers(gripper, joint_names_prefix)
        if ft_sensor:
            self._init_ft_sensor()

### private methods ###

    def _init_controllers(self, gripper, joint_names_prefix=None):
        traj_publisher = JOINT_PUBLISHER_ROBOT
        self.joint_names = None if joint_names_prefix is None else get_arm_joint_names(joint_names_prefix)

        # Flexible trajectory (point by point)

        traj_publisher_flex = self.ns + '/' + traj_publisher + '/command'

        cprint.blue("connecting to: {}".format(traj_publisher_flex))

        self._flex_trajectory_pub = rospy.Publisher(traj_publisher_flex,
                                                    JointTrajectory,
                                                    queue_size=10)

        self.joint_traj_controller = JointTrajectoryController(
            publisher_name=traj_publisher, namespace=self.ns, joint_names=self.joint_names)

        self.gripper = None
        if gripper:
            self.gripper = GripperController(namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)

    def _init_ik_solver(self):
        self.kdl = ur_kinematics(self._robot_urdf, base_link=self.base_link, ee_link=self.ee_link, prefix=self.joint_names_prefix, rospackage=self._robot_urdf_package)
        
        if self.ik_solver == IKFAST:
            # IKfast libraries
            if self._robot_urdf == 'ur3_robot':
                self.arm_ikfast = ur_ikfast.URKinematics('ur3')
            elif self._robot_urdf == 'ur3e_robot':
                self.arm_ikfast = ur_ikfast.URKinematics('ur3e')
        elif self.ik_solver == TRAC_IK:
            try:
                if not rospy.has_param("robot_description"):
                    self.trac_ik = IK(base_link=self.base_link, tip_link=self.ee_link, solve_type="Distance", 
                                      urdf_string=utils.load_urdf_string(self._robot_urdf_package, self._robot_urdf))
                else:
                    self.trac_ik = IK(base_link=self.base_link, tip_link=self.ee_link, solve_type="Distance")
            except Exception as e:
                rospy.logerr("Could not instantiate TRAC_IK" + str(e))
        else:
            raise Exception("unsupported ik_solver", self.ik_solver)

    def _init_ft_sensor(self):
        # Publisher of wrench
        namespace = '' if self.ns is None else self.ns
        print("publish filtered wrench:", '%s/%s/filtered' % (namespace, self.ft_topic))
        self.pub_ee_wrench = rospy.Publisher('%s/%s/filtered' % (namespace, self.ft_topic),
                                             Wrench,
                                             queue_size=50)

        self.ft_sensor = FTsensor(namespace='%s/%s' % (namespace, self.ft_topic))
        self.set_wrench_offset(override=False)

    def _update_wrench_offset(self):
        namespace = '' if self.ns is None else self.ns
        self.wrench_offset = self.get_filtered_ft().tolist()
        rospy.set_param('%s/ft_offset' % namespace, self.wrench_offset)

    def _flexible_trajectory(self, position, time=5.0, vel=None):
        """ Publish point by point making it more flexible for real-time control """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = JOINT_ORDER if self.joint_names is None else self.joint_names

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
            inv_ee_transform[3:] = transformations.quaternion_inverse(inv_ee_transform[3:])
            pose = np.array(conversions.transform_end_effector(pose, inv_ee_transform))

        if self.ik_solver == IKFAST:
            ik = self.arm_ikfast.inverse(pose, q_guess=self.joint_angles())

        elif self.ik_solver == TRAC_IK:
            ik = self.trac_ik.get_ik(self.joint_angles(), pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])
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
            namespace = 'ur3' if self.ns is None else self.ns
            self.wrench_offset = rospy.get_param('/%s/ft_offset' % namespace, None)
            if self.wrench_offset is None:
                self._update_wrench_offset()

    def get_ee_wrench_hist(self, hist_size=24):
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")

        q_hist = self.joint_traj_controller.get_joint_positions_hist()[:hist_size]
        ft_hist = self.ft_sensor.get_filtered_wrench(hist_size=hist_size)

        if self.wrench_offset is not None:
            ft_hist = np.array(ft_hist) - np.array(self.wrench_offset)

        poses_hist = [self.end_effector(q) for q in q_hist]
        wrench_hist = [spalg.convert_wrench(wft, p).tolist() for p, wft in zip(poses_hist, ft_hist)]

        return np.array(wrench_hist)

    def get_ee_wrench(self, relative=True):
        """ Compute the wrench (force/torque) in task-space """
        if self.ft_sensor is None:
            return np.zeros(6)

        wrench_force = self.ft_sensor.get_filtered_wrench()

        if self.wrench_offset is not None:
            wrench_force = np.array(wrench_force) - np.array(self.wrench_offset)

        # compute force transformation?
        # # # Transform of EE
        pose = self.end_effector()
        ee_wrench_force = spalg.convert_wrench(wrench_force, pose)
        
        if relative:
            return ee_wrench_force
        else:
            return wrench_force

    def publish_wrench(self, relative=False):
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")

        " Publish arm's end-effector wrench "
        wrench = self.get_ee_wrench(relative)
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
        self.joint_traj_controller.start(delay=0.01, wait=wait)
        self.joint_traj_controller.clear_points()

    def set_joint_positions_flex(self, position, t=5.0, v=None):
        qc = self.joint_angles()
        deltaq = (qc - position)
        speed = deltaq / t
        cmd = position
        if np.any(np.abs(speed) > (self.max_joint_speed/t)):
            rospy.logdebug("Attempting to exceeded max speed %s, ignoring command" % speed)
            return SPEED_LIMIT_EXCEEDED
        self._flexible_trajectory(cmd, t, v)
        return DONE

    def set_target_pose(self, pose, wait=False, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self._solve_ik(pose)
        if q is None:
            # IK not found
            return IK_NOT_FOUND
        else:
            return self.set_joint_positions(q, wait=wait, t=t)

    def set_target_pose_flex(self, pose, t=5.0):
        """ Supported pose is only x y z aw ax ay az """
        q = self._solve_ik(pose)
        if q is None:
            # IK not found
            return IK_NOT_FOUND
        else:
            return self.set_joint_positions_flex(q, t=t)
