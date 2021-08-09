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
    BASE_LINK, EE_LINK, FT_LINK

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
                 robot_urdf='ur3e',
                 robot_urdf_package=None,
                 ik_solver=TRAC_IK,
                 namespace='',
                 gripper=False,
                 joint_names_prefix=None,
                 ft_topic=None,
                 base_link=None,
                 ee_link=None,
                 ft_link=None):
        """ ft_sensor bool: whether or not to try to load ft sensor information
            ee_transform array [x,y,z,ax,ay,az,w]: optional transformation to the end-effector
                                                  that is applied before doing any operation in task-space
            robot_urdf string: name of the robot urdf file to be used
            namespace string: nodes namespace prefix
            gripper bool: enable gripper control
        """

        cprint.ok("ft_sensor: {}, ee_link: {}, \n robot_urdf: {}".format(ft_sensor, ee_link, robot_urdf))

        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._current_ft = []

        self._robot_urdf = robot_urdf
        self._robot_urdf_package = robot_urdf_package if robot_urdf_package is not None else 'ur_pykdl'

        self.ft_sensor = None
        self.ft_topic = ft_topic if ft_topic is not None else FT_SUBSCRIBER

        self.ik_solver = ik_solver
        
        assert namespace is not None, "namespace cannot be None"
        self.ns = namespace
        self.joint_names_prefix = joint_names_prefix

        _base_link = base_link if base_link is not None else BASE_LINK
        _ee_link = ee_link if ee_link is not None else EE_LINK
        _ft_frame = ft_link if ft_link is not None else FT_LINK

        self.base_link = _base_link if joint_names_prefix is None else joint_names_prefix + _base_link
        self.ee_link = _ee_link if joint_names_prefix is None else joint_names_prefix + _ee_link
        self.ft_frame = _ft_frame if joint_names_prefix is None else joint_names_prefix + _ft_frame

        # self.max_joint_speed = np.deg2rad([100, 100, 100, 200, 200, 200]) # deg/s -> rad/s
        self.max_joint_speed = np.deg2rad([191, 191, 191, 371, 371, 371])

        self._init_ik_solver(self.base_link, self.ee_link)
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
            publisher_name=traj_publisher, namespace=self.ns, joint_names=self.joint_names, timeout=10.0)

        self.gripper = None
        if gripper:
            self.gripper = GripperController(namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)

    def _init_ik_solver(self, base_link, ee_link):
        self.base_link = base_link
        self.ee_link = ee_link
        if rospy.has_param("robot_description"):
            self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link)
        else:
            self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link, robot=self._robot_urdf, prefix=self.joint_names_prefix, rospackage=self._robot_urdf_package)

        if self.ik_solver == IKFAST:
            # IKfast libraries
            try:
                self.arm_ikfast = ur_ikfast.URKinematics(self._robot_urdf)
            except Exception:
                rospy.logerr("IK solver set to IKFAST but no ikfast found for: %s. Switching to TRAC_IK" % self._robot_urdf)
                self.ik_solver == TRAC_IK
                return self._init_ik_solver(base_link, ee_link)
        elif self.ik_solver == TRAC_IK:
            try:
                if not rospy.has_param("robot_description"):
                    self.trac_ik = IK(base_link=base_link, tip_link=ee_link, solve_type="Distance", timeout=0.002, epsilon=1e-5,
                                      urdf_string=utils.load_urdf_string(self._robot_urdf_package, self._robot_urdf))
                else:
                    self.trac_ik = IK(base_link=base_link, tip_link=ee_link, solve_type="Distance")
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

    def _solve_ik(self, pose, q_guess=None, attempts=5, verbose=True):
        q_guess_ = q_guess if q_guess is not None else self.joint_angles()
        # TODO(cambel): weird it shouldn't happen but...
        if isinstance(q_guess, np.float64):
            q_guess_ = None

        if self.ik_solver == IKFAST:
            ik = self.arm_ikfast.inverse(pose, q_guess=q_guess_)

        elif self.ik_solver == TRAC_IK:
            ik = self.trac_ik.get_ik(q_guess_, *pose)
            if ik is None:
                if attempts > 0:
                    return self._solve_ik(pose, q_guess, attempts-1)
                if verbose:
                    rospy.logwarn("TRACK-IK: solution not found!")

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

    def get_ee_wrench(self):
        """ Compute the wrench (force/torque) in task-space """
        if self.ft_sensor is None:
            return np.zeros(6)

        wrench_force = self.ft_sensor.get_filtered_wrench()

        if self.wrench_offset is not None:
            wrench_force = np.array(wrench_force) - np.array(self.wrench_offset)

        # compute force transformation?
        # # # Transform of EE
        pose = self.end_effector(tip_link=self.ft_frame)

        ee_wrench_force = spalg.convert_wrench(wrench_force, pose)

        return ee_wrench_force

    def publish_wrench(self):
        if self.ft_sensor is None:
            raise Exception("FT Sensor not initialized")

        " Publish arm's end-effector wrench "
        wrench = self.get_ee_wrench()
        # Note you need to call rospy.init_node() before this will work
        self.pub_ee_wrench.publish(conversions.to_wrench(wrench))

    def end_effector(self,
                     joint_angles=None,
                     rot_type='quaternion',
                     tip_link=None):
        """ Return End Effector Pose """

        joint_angles = self.joint_angles() if joint_angles is None else joint_angles

        if rot_type == 'quaternion':
            # forward kinematics
            return self.kdl.forward(joint_angles, tip_link)

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
        return DONE

    def set_joint_trajectory(self, trajectory, velocities=None, accelerations=None, t=5.0):
        dt = float(t)/float(len(trajectory))

        vel = None
        acc = None

        if velocities is not None:
            vel = [velocities] * 6
        if accelerations is not None:
            acc = [accelerations] * 6

        for i, q in enumerate(trajectory):
            self.joint_traj_controller.add_point(positions=q,
                                                 time=(i+1) * dt,
                                                 velocities=vel,
                                                 accelerations=acc)
        self.joint_traj_controller.start(delay=0.01, wait=True)
        self.joint_traj_controller.clear_points()

    def set_joint_positions_flex(self, position, t=5.0, v=None):
        qc = self.joint_angles()
        deltaq = (qc - position)
        speed = deltaq / t
        cmd = position
        if np.any(np.abs(speed) > self.max_joint_speed):
            rospy.logwarn("Exceeded max speed %s deg/s, ignoring command" % np.round(np.rad2deg(speed), 0))
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

### Complementary control methods ###

    def move_relative(self, delta, relative_to_ee=False, wait=True, t=5.):
        """
            Move relative to the current pose of the robot
            delta: array[6], translations and rotations(euler angles) from the current pose
            relative_to_ee: bool, whether to consider the delta relative to the robot's base or its end-effector (TCP)
            wait: bool, wait for the motion to be completed
            t: float, duration of the motion (how fast it will be)
        """
        cpose = self.end_effector()
        cmd = transformations.pose_euler_to_quaternion(cpose, delta, ee_rotation=relative_to_ee)
        return self.set_target_pose(cmd, wait=True, t=t)

    def move_linear(self, pose, eef_step=0.01, t=5.0):
        """
            CAUTION: simple linear interpolation
            pose: array[7], target translation and rotation
            granularity: int, number of point for the interpolation
            t: float, duration in seconds
        """
        joint_trajectory = self.compute_cartesian_path(pose, eef_step, t)
        self.set_joint_trajectory(joint_trajectory, t=t)

    def compute_cartesian_path(self, pose, eef_step=0.01, t=5.0):
        """
            CAUTION: simple linear interpolation
            pose: array[7], target translation and rotation
            granularity: int, number of point for the interpolation
            t: float, duration in seconds
        """
        cpose = self.end_effector()
        translation_dist = np.linalg.norm(cpose[:3])
        rotation_dist = Quaternion.distance(transformations.vector_to_pyquaternion(cpose[3:]), transformations.vector_to_pyquaternion(pose[3:])) / 2.0

        steps = int((translation_dist + rotation_dist) / eef_step)

        points = np.linspace(cpose[:3], pose[:3], steps)
        rotations = Quaternion.intermediates(transformations.vector_to_pyquaternion(cpose[3:]), transformations.vector_to_pyquaternion(pose[3:]), steps, include_endpoints=True)

        joint_trajectory = []

        for i, (point, rotation) in enumerate(zip(points, rotations)):
            cmd = np.concatenate([point, transformations.vector_from_pyquaternion(rotation)])
            q_guess = None if i < 2 else np.mean(joint_trajectory[:-1], 0)
            q = self._solve_ik(cmd, q_guess)
            if q is not None:  # ignore points with no IK solution, can we do better?
                joint_trajectory.append(q)

        dt = t/float(len(joint_trajectory))
        # TODO(cambel): is this good enough to catch big jumps due to IK solutions?
        return spalg.jump_threshold(np.array(joint_trajectory), dt, 2.5)
