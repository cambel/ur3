# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import collections
import numpy as np

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from geometry_msgs.msg import WrenchStamped

from ur_control import utils, spalg, conversions, transformations
from ur_control.constants import JOINT_ORDER, JOINT_TRAJECTORY_CONTROLLER, FT_SUBSCRIBER, IKFAST, TRAC_IK, \
    DONE, SPEED_LIMIT_EXCEEDED, IK_NOT_FOUND, get_arm_joint_names, \
    BASE_LINK, EE_LINK

from std_srvs.srv import Empty, SetBool

try:
    from ur_ikfast import ur_kinematics as ur_ikfast
except ImportError:
    print("Import ur_ikfast not available, IKFAST would not be supported without it")

from ur_control.controllers_connection import ControllersConnection
from ur_control.controllers import JointTrajectoryController, FTsensor, GripperController
from ur_pykdl import ur_kinematics
from trac_ik_python.trac_ik import IK as TRACK_IK_SOLVER

cprint = utils.TextColors()


def resolve_parameter(value, default_value):
    if value:
        return value
    if default_value:
        return default_value
    raise ValueError("No value defined for parameter")


class Arm(object):
    """ UR3 arm controller """

    def __init__(self,
                 robot_urdf='ur3e',
                 robot_urdf_package=None,
                 ik_solver=TRAC_IK,
                 namespace=None,
                 gripper=False,
                 joint_names_prefix=None,
                 ft_topic=None,
                 base_link=None,
                 ee_link=None):
        """ ee_transform array [x,y,z,ax,ay,az,w]: optional transformation to the end-effector
                                                  that is applied before doing any operation in task-space
            robot_urdf string: name of the robot urdf file to be used
            namespace string: nodes namespace prefix
            gripper bool: enable gripper control
        """

        cprint.ok("ft_sensor: {}, ee_link: {}, \n robot_urdf: {}".format(bool(ft_topic), ee_link, robot_urdf))

        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()

        self.current_ft_value = np.zeros(6)
        self.wrench_queue = collections.deque(maxlen=25)  # store history of FT data

        self._robot_urdf = robot_urdf
        self._robot_urdf_package = robot_urdf_package if robot_urdf_package is not None else 'ur_pykdl'

        self.ft_topic = resolve_parameter(ft_topic, FT_SUBSCRIBER)

        self.ik_solver = ik_solver

        self.ns = resolve_parameter(namespace, "")

        base_link = resolve_parameter(base_link, BASE_LINK)
        ee_link = resolve_parameter(ee_link, EE_LINK)

        # Support for joint prefixes
        self.joint_names_prefix = joint_names_prefix
        self.base_link = base_link if joint_names_prefix is None else joint_names_prefix + base_link
        self.ee_link = ee_link if joint_names_prefix is None else joint_names_prefix + ee_link

        # self.max_joint_speed = np.deg2rad([100, 100, 100, 200, 200, 200]) # deg/s -> rad/s
        self.max_joint_speed = np.deg2rad([191, 191, 191, 371, 371, 371])

        self._init_ik_solver(self.base_link, self.ee_link)
        self._init_controllers(gripper, joint_names_prefix)
        if ft_topic:
            self._init_ft_sensor()

        self.controller_manager = ControllersConnection(namespace)

### private methods ###

    def _init_controllers(self, gripper, joint_names_prefix=None):
        traj_publisher = JOINT_TRAJECTORY_CONTROLLER
        self.joint_names = None if joint_names_prefix is None else get_arm_joint_names(joint_names_prefix)

        # Flexible trajectory (point by point)

        traj_publisher_flex = self.ns + '/' + traj_publisher + '/command'
        cprint.blue("connecting to: {}".format(traj_publisher_flex))
        self._flex_trajectory_pub = rospy.Publisher(traj_publisher_flex, JointTrajectory, queue_size=10)

        self.joint_traj_controller = JointTrajectoryController(
            publisher_name=traj_publisher, namespace=self.ns, joint_names=self.joint_names, timeout=1.0)

        self.gripper = None
        if gripper:
            self.gripper = GripperController(namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)

    def _init_ik_solver(self, base_link, ee_link):
        # Instantiate KDL kinematics solver to compute forward kinematics
        if rospy.has_param("robot_description"):
            self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link)
        else:
            self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link, robot=self._robot_urdf, prefix=self.joint_names_prefix, rospackage=self._robot_urdf_package)

        # Instantiate Inverse kinematics solver
        if self.ik_solver == IKFAST:
            # IKfast libraries
            try:
                self.arm_ikfast = ur_ikfast.URKinematics(self._robot_urdf)
            except Exception:
                raise ValueError("IK solver set to IKFAST but no ikfast found for: %s. " % self._robot_urdf)
        elif self.ik_solver == TRAC_IK:
            try:
                if not rospy.has_param("robot_description"):
                    self.trac_ik = TRACK_IK_SOLVER(base_link=base_link, tip_link=ee_link, solve_type="Distance", timeout=0.002, epsilon=1e-5,
                                                   urdf_string=utils.load_urdf_string(self._robot_urdf_package, self._robot_urdf))
                else:
                    self.trac_ik = TRACK_IK_SOLVER(base_link=base_link, tip_link=ee_link, solve_type="Distance")
            except Exception as e:
                rospy.logerr("Could not instantiate TRAC_IK" + str(e))
        else:
            raise Exception("unsupported ik_solver", self.ik_solver)

    def _init_ft_sensor(self):
        # Publisher of wrench
        ft_namespace = '%s/%s/filtered' % (self.ns, self.ft_topic)
        rospy.Subscriber(ft_namespace, WrenchStamped, self.__ft_callback__)

        self._zero_ft = rospy.ServiceProxy('%s/%s/zero_ftsensor' % (self.ns, self.ft_topic), Empty)
        self._zero_ft.wait_for_service(rospy.Duration(2.0))
        self._ft_filtered = rospy.ServiceProxy('%s/%s/enable_filtering' % (self.ns, self.ft_topic), SetBool)
        self._ft_filtered.wait_for_service(rospy.Duration(1.0))

        # Check that the FT topic is publishing
        if not utils.wait_for(lambda: self.current_ft_value is not None, timeout=2.0):
            rospy.logerr('Timed out waiting for {0} topic'.format(ft_namespace))
            return

    def __ft_callback__(self, msg):
        self.current_ft_value = conversions.from_wrench(msg.wrench)
        self.wrench_queue.append(self.current_ft_value)

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
        """ Get measurements from FT Sensor in its default frame of reference.
            Measurements are filtered with a low-pass filter.
            Measurements are given in sensors orientation.
        """
        if self.current_ft_value is None:
            raise Exception("FT Sensor not initialized")

        ft_limitter = [300, 300, 300, 30, 30, 30]  # Enforce measurement limits (simulation)
        ft = self.current_ft_value
        ft = [
            ft[i] if abs(ft[i]) < ft_limitter[i] else ft_limitter[i]
            for i in range(6)
        ]
        return np.array(ft)

    def get_ee_wrench_hist(self, hist_size=24):
        if self.current_ft_value is None:
            raise Exception("FT Sensor not initialized")

        q_hist = self.joint_traj_controller.get_joint_positions_hist()[:hist_size]
        ft_hist = np.array(self.wrench_queue)[:hist_size]

        poses_hist = [self.end_effector(q) for q in q_hist]
        wrench_hist = [spalg.convert_wrench(wft, p).tolist() for p, wft in zip(poses_hist, ft_hist)]

        return np.array(wrench_hist)

    def get_ee_wrench(self, hand_frame_control=False):
        """ Compute the wrench (force/torque) in task-space """
        if self.current_ft_value is None:
            return np.zeros(6)

        wrench_force = self.current_ft_value
        if not hand_frame_control:
            return wrench_force
        else:
            # Transform force to end effector frame
            pose = self.end_effector()
            ee_wrench_force = spalg.convert_wrench(wrench_force, pose)

            return ee_wrench_force

    def zero_ft_sensor(self):
        self._zero_ft()

    def set_ft_filtering(self, active=True):
        self._ft_filtered(active)

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
        speed = (qc - position) / t
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

    def move_relative(self, transformation, relative_to_tcp=True, duration=5.0, wait=True):
        """ Move end-effector backwards relative to its position in a straight line """
        new_pose = transformations.transform_pose(self.end_effector(), transformation, rotated_frame=relative_to_tcp)
        self.set_target_pose(pose=new_pose, t=duration, wait=wait)
