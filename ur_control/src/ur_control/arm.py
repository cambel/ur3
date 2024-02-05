# The MIT License (MIT)
#
# Copyright (c) 2018-2023 Cristian Beltran
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
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Empty, SetBool, Trigger

from ur_control import utils, spalg, conversions, transformations
from ur_control.exceptions import InverseKinematicsException
from ur_control.controllers_connection import ControllersConnection
from ur_control.controllers import JointTrajectoryController
from ur_control.grippers import GripperController, RobotiqGripper
from ur_control.constants import BASE_LINK, EE_LINK, JOINT_TRAJECTORY_CONTROLLER, FT_SUBSCRIBER,  \
    ExecutionResult, IKSolverType, GripperType, \
    get_arm_joint_names
from ur_control.ur_services import URServices

try:
    from ur_ikfast import ur_kinematics as ur_ikfast
except ImportError:
    print("Import ur_ikfast not available, IKFAST would not be supported without it")
from ur_pykdl import ur_kinematics
from trac_ik_python.trac_ik import IK as TRACK_IK_SOLVER

cprint = utils.TextColors()


class Arm(object):
    """ Universal Robots arm controller """

    def __init__(self,
                 namespace: str = None,
                 ik_solver: IKSolverType = IKSolverType.TRAC_IK,
                 gripper_type: GripperType = GripperType.GENERIC,
                 ft_topic: str = None,
                 base_link: str = None,
                 ee_link: str = None,
                 joint_names_prefix: str = None):
        """ 

        Parameters
        ----------
        namespace : optional
            ROS namespace of the robot e.g., '/ns/robot_description'
        ik_solver : optional
            inverse kinematic solver to be used
        gripper_type : optional
            gripper control approach. Generic or specific for real Robotiq gripper
        ft_topic: optional
            topic from which to read the wrench
        base_link: optional
            robot base frame. Excluding the prefix used in 'joint_names_prefix'
        ee_link: optional
            end-effector frame. Excluding the prefix used in 'joint_names_prefix'
        joint_names_prefix: optional
            optionally specify a prefix when multiple robots are defined. For example,
            if 'a_bot' is defined, all joints will be consider like 'a_bot_link_name'

        Raises
        ------
        ValueError
            If a non-supported gripper type or ik solver is defined

        """

        self.ns = utils.solve_namespace(namespace)

        base_link = utils.resolve_parameter(value=base_link, default_value=BASE_LINK)
        ee_link = utils.resolve_parameter(value=ee_link, default_value=EE_LINK)

        # Support for joint prefixes
        self.joint_names_prefix = utils.resolve_parameter(joint_names_prefix, '')
        self.base_link = base_link if joint_names_prefix is None else joint_names_prefix + base_link
        self.ee_link = ee_link if joint_names_prefix is None else joint_names_prefix + ee_link

        self.ik_solver = ik_solver

        self.ft_topic = utils.resolve_parameter(value=ft_topic, default_value=FT_SUBSCRIBER)
        self.current_ft_value = np.zeros(6)
        self.wrench_queue = collections.deque(maxlen=25)  # store history of FT data

        # self.max_joint_speed = np.deg2rad([100, 100, 100, 200, 200, 200]) # deg/s -> rad/s
        self.max_joint_speed = np.deg2rad([191, 191, 191, 371, 371, 371])

        cprint.ok("Initializing ur robot with parameters")
        cprint.ok("gripper: {}, ft_sensor_topic: {}, \nbase_link: {}, ee_link: {}"
                  .format(gripper_type, self.ft_topic, self.base_link, self.ee_link))

        self.__init_controllers__(gripper_type, joint_names_prefix)
        self.__init_ik_solver__(self.base_link, self.ee_link)

        self.__init_ft_sensor__()

        self.controller_manager = ControllersConnection(self.ns)
        self.dashboard_services = URServices(self.ns)

### private methods ###

    def __init_controllers__(self, gripper_type, joint_names_prefix=None):
        self.joint_names = None if joint_names_prefix is None else get_arm_joint_names(joint_names_prefix)

        self.joint_traj_controller = JointTrajectoryController(publisher_name=JOINT_TRAJECTORY_CONTROLLER,
                                                               namespace=self.ns,
                                                               joint_names=self.joint_names,
                                                               timeout=1.0)

        self.gripper = None

        if not gripper_type:
            rospy.logwarn("Loading without gripper")
            return

        if gripper_type == GripperType.GENERIC:
            self.gripper = GripperController(namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)
        elif gripper_type == GripperType.ROBOTIQ:
            self.gripper = RobotiqGripper(namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)
        else:
            raise ValueError("Invalid gripper type %s" % gripper_type)

    def __init_ik_solver__(self, base_link, ee_link):
        # Instantiate KDL kinematics solver to compute forward kinematics
        if rospy.has_param("robot_description"):
            self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link)
        else:
            raise ValueError("robot_description not found in the parameter server")

        # Instantiate Inverse kinematics solver
        if self.ik_solver == IKSolverType.IKFAST:
            # IKfast libraries
            try:
                # TODO use the parameter robot_description
                self.arm_ikfast = ur_ikfast.URKinematics(self._robot_urdf)
            except Exception:
                raise ValueError("IK solver set to IKFAST but no ikfast found for: %s. " % self._robot_urdf)
        elif self.ik_solver == IKSolverType.TRAC_IK:
            try:
                self.trac_ik = TRACK_IK_SOLVER(base_link=base_link, tip_link=ee_link, solve_type="Distance")
            except Exception as e:
                rospy.logerr("Could not instantiate TRAC_IK" + str(e))
        elif self.ik_solver == IKSolverType.KDL:
            pass
        else:
            raise Exception("unsupported ik_solver", self.ik_solver)

    def __init_ft_sensor__(self):
        # Publisher of wrench
        ft_namespace = self.ns + self.ft_topic + '/filtered'
        if not utils.topic_exist(ft_namespace):
            rospy.logwarn("Filtered FT topic not found. Using raw sensor directly.")
            # Try the raw FT topic
            ft_namespace = self.ns + self.ft_topic
            rospy.Subscriber(ft_namespace, WrenchStamped, self.__ft_callback__)
            self._zero_ft_filtered = lambda: None
            self._ft_filtered = lambda: None
        else:
            rospy.Subscriber(ft_namespace, WrenchStamped, self.__ft_callback__)

            self._zero_ft_filtered = rospy.ServiceProxy('%s/%s/filtered/zero_ftsensor' % (self.ns, self.ft_topic), Empty)
            self._zero_ft_filtered.wait_for_service(rospy.Duration(2.0))

            if not rospy.has_param("use_gazebo_sim"):
                self._zero_ft = rospy.ServiceProxy('%s/ur_hardware_interface/zero_ftsensor' % self.ns, Trigger)
                self._zero_ft.wait_for_service(rospy.Duration(2.0))

            self._ft_filtered = rospy.ServiceProxy('%s/%s/filtered/enable_filtering' % (self.ns, self.ft_topic), SetBool)
            self._ft_filtered.wait_for_service(rospy.Duration(1.0))

            # Check that the FT topic is publishing
            if not utils.wait_for(lambda: self.current_ft_value is not None, timeout=2.0):
                rospy.logerr('Timed out waiting for {0} topic'.format(ft_namespace))

    def __ft_callback__(self, msg):
        self.current_ft_value = conversions.from_wrench(msg.wrench)
        self.wrench_queue.append(self.current_ft_value)

### Data access methods ###

    def inverse_kinematics(self,
                           pose: np.ndarray,
                           seed: np.ndarray = None,
                           attempts: int = 0,
                           verbose: bool = True) -> np.ndarray:
        """
        return a joint configuration for a given Cartesian pose of the end-effector
        (ee_link) if any.

        Parameters
        ----------
        pose : 
            Cartesian pose of the end-effector defined as ee_link
        seed : optional
            if given, attempt to return a joint configuration closer to the seed
        attempts : int, optional
            number of attempts to find a IK solution. It may be useful for sample
            based solvers such as TRAC-IK. It would not change the result of an
            analytical solvers such as IKFast.
        verbose : bool, optional
            print a warning message when IK solutions are not found

        Returns
        -------
        res : ndarray
            Joint configuration if any or None

        Raises
        ------
        ValueError
            If the rot_type is different from 'quaternion' or 'euler'
        """
        q_guess_ = seed if seed is not None else self.joint_angles()

        if self.ik_solver == IKSolverType.IKFAST:
            # TODO: transform pose to the default tip used by IKFast (tool0)
            ik = self.arm_ikfast.inverse(pose, q_guess=q_guess_)
        elif self.ik_solver == IKSolverType.TRAC_IK:
            ik = self.trac_ik.get_ik(q_guess_, *pose)
        elif self.ik_solver == IKSolverType.KDL:
            ik = self.kdl.inverse_kinematics(pose[:3], pose[3:], seed=q_guess_)

        if ik is None:
            if attempts > 0:
                return self.inverse_kinematics(pose, seed, attempts-1)
            if verbose:
                rospy.logwarn(f"{self.ik_solver}: solution not found!")
            raise InverseKinematicsException(f"{self.ik_solver}: solution not found!")
        return ik

    def end_effector(self,
                     joint_angles=None,
                     rot_type='quaternion',
                     tip_link=None) -> np.ndarray:
        """ 
        Return the Cartesian pose of the end-effector in the robot base frame (base_link).

        Parameters
        ----------
        joint_angles : ndarray, optional
            If not given, the current joint configuration will be used.
            If provided, the joint configuration is expected in the order given by constants.JOINT_ORDER
        rot_type : str, optional
            Rotation representation to be returned.
            Valid types "quaternion" or "euler"
        tip_link : str, optional
            Return the Cartesian pose of the tip_link if provided.
            Otherwise, use the default ee_link

        Returns
        -------
        res : ndarray
            The Cartesian pose in the form of 
            quaternion: [x, y, z, aw, ax, ay, az] or
            euler: [x, y, z, roll, pitch, yaw]
            in radians.

        Raises
        ------
        ValueError
            If the rot_type is different from 'quaternion' or 'euler'
        """

        joint_angles = self.joint_angles() if joint_angles is None else joint_angles

        if rot_type == 'quaternion':
            # forward kinematics
            return self.kdl.forward(joint_angles, tip_link)

        elif rot_type == 'euler':
            x = self.end_effector(joint_angles, tip_link=tip_link)
            euler = np.array(transformations.euler_from_quaternion(x[3:], axes='sxyz'))
            return np.concatenate((x[:3], euler))

        else:
            raise ValueError("Rotation Type not supported", rot_type)

    def joint_angle(self, joint: str) -> float:
        """
        Return the requested joint angle in radians.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_positions()[joint_idx]

    def joint_angles(self) -> np.ndarray:
        """
        Returns the current joint positions in radians and 
        in the order given by constants.JOINT_ORDER.
        """
        return self.joint_traj_controller.get_joint_positions()

    def joint_velocity(self, joint: str) -> float:
        """
        Return the requested joint velocity in radians/secs.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_velocities()[joint_idx]

    def joint_velocities(self) -> np.ndarray:
        """
        Returns the current joint velocities.
        """
        return self.joint_traj_controller.get_joint_velocities()

    def joint_effort(self, joint: str) -> float:
        """
        Return the requested joint effort.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_efforts()[joint_idx]

    def joint_efforts(self) -> np.ndarray:
        """
        Returns the current joint efforts.
        """
        return self.joint_traj_controller.get_joint_efforts()

    def get_wrench_history(self, hist_size=24, hand_frame_control=False):
        if self.current_ft_value is None:
            raise Exception("FT Sensor not initialized")

        ft_hist = np.array(self.wrench_queue)[:hist_size]

        if hand_frame_control:
            q_hist = self.joint_traj_controller.get_joint_positions_hist()[:hist_size]
            poses_hist = [self.end_effector(q, tip_link=self.ee_link) for q in q_hist]
            wrench_hist = [spalg.convert_wrench(wft, p).tolist() for p, wft in zip(poses_hist, ft_hist)]
        else:
            wrench_hist = ft_hist

        return np.array(wrench_hist)

    def get_wrench(self,
                   base_frame_control=False,
                   hand_frame_control=False) -> np.ndarray:
        """ 
        Returns the wrench (force/torque) in task-space.
        By default, return the wrench as read from the sensor topic.

        Parameters
        ----------
        base_frame_control : bool, optional
            If True, returns the wrench with respect to the robot base frame
        hand_frame_control : bool, optional
            If True, returns the wrench with respect to the end-effector frame
            If both base_frame_control and hand_frame_control are set to True.
            the former is considered.
        Returns
        -------
        res : np.ndarray
            Returns the wrench in the requested frame
        """
        if self.current_ft_value is None:
            # No values have been received from sensor's topic
            return np.zeros(6)

        wrench_force = self.current_ft_value
        if not hand_frame_control and not base_frame_control:
            return wrench_force

        if base_frame_control:
            # Transform force/torque from sensor to robot base frame
            transform = self.end_effector(tip_link=self.joint_names_prefix + "wrist_3_link")
            ee_wrench_force = spalg.convert_wrench(wrench_force, transform)

            return ee_wrench_force
        else:
            # Transform force/torque from sensor to end effector frame
            transform = self.end_effector(tip_link=self.ee_link)
            ee_wrench_force = spalg.convert_wrench(wrench_force, transform)

            return ee_wrench_force

### Control Methods ###

    def set_joint_positions(self,
                            target_time: float,
                            positions: np.ndarray,
                            velocities: np.ndarray = None,
                            accelerations: np.ndarray = None,
                            wait: bool = False) -> ExecutionResult:
        """
        Run the joint trajectory controller towards a single waypoint starting now.

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the 
            duration of the trajectory.
        positions : numpy.ndarray
            target joint configuration in the order given by constants.JOINT_ORDER
        velocities : numpy.ndarray, optional
            target joint velocities
        accelerations : numpy.ndarray, optional
            target joint accelerations
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is succesful when waiting for the execution to be 
            completed. Otherwise returns true if the trajectory was started.
        """
        self.joint_traj_controller.add_point(positions=positions,
                                             velocities=velocities,
                                             accelerations=accelerations,
                                             target_time=target_time)
        if wait:
            self.joint_traj_controller.start(delay=0, wait=True)
        else:
            self.joint_traj_controller.start_no_action_server()

        # Always clear the trajectory goal
        self.joint_traj_controller.clear_points()

        if wait:
            res = self.joint_traj_controller.get_result()
            return ExecutionResult.DONE if res.error_code == 0 else ExecutionResult.CONTROLLER_FAILED
        return ExecutionResult.DONE

    def set_joint_trajectory(self,
                             target_time: float,
                             trajectory: np.ndarray,
                             velocities: np.ndarray = None,
                             accelerations: np.ndarray = None) -> ExecutionResult:
        """
        Start the joint trajectory controller with a multi-waypoint trajectory.

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the 
            duration of the trajectory.
        positions : 2-D numpy.ndarray
            list of target joint configuration for each waypoint in the order given by constants.JOINT_ORDER
        velocities : 2-D numpy.ndarray, optional
            list of target joint velocities for each waypoint
        accelerations : 2-D numpy.ndarray, optional
            list of target joint accelerations for each waypoint
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is succesfully executed.
        """
        dt = target_time/len(trajectory)

        for i, q in enumerate(trajectory):
            self.joint_traj_controller.add_point(positions=q,
                                                 target_time=(i+1) * dt,
                                                 velocities=velocities,
                                                 accelerations=accelerations)
        self.joint_traj_controller.start(delay=0, wait=True)
        self.joint_traj_controller.clear_points()

        res = self.joint_traj_controller.get_result()
        return ExecutionResult.DONE if res.error_code == 0 else ExecutionResult.CONTROLLER_FAILED

    def set_target_pose(self,
                        target_time: float,
                        pose: np.ndarray,
                        wait: bool = False) -> ExecutionResult:
        """
        Reach a given target cartesian pose with the end-effector (ee_link)

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the 
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is succesfully executed.
        """
        q = self.inverse_kinematics(pose)
        if q is None:
            rospy.logdebug("IK not found")
            raise InverseKinematicsException("IK solver failed to find a solution")
        else:
            return self.set_joint_positions(positions=q, target_time=target_time, wait=wait)

    def set_pose_trajectory(self,
                            target_time: float,
                            trajectory: np.ndarray) -> ExecutionResult:
        """
        Reach a given target cartesian pose with the end-effector (ee_link)

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the 
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool or str
            True if the trajectory is succesfully executed.
            If the IK solver fails, return "ik_not_found"
        """
        joint_trajectory = []
        previous_q = self.joint_angles()
        for i, pose in enumerate(trajectory):
            q = self.inverse_kinematics(pose, seed=previous_q)
            if q is None:
                raise InverseKinematicsException("IK solver failed to find a solution")

            previous_q = q
            joint_trajectory.append(q)
        return self.set_joint_trajectory(trajectory=joint_trajectory, target_time=target_time)

    def move_relative(self,
                      target_time: float,
                      transformation: np.array,
                      relative_to_tcp: bool = True,
                      wait: bool = True) -> ExecutionResult:
        """ 
        Move end-effector (ee_link) relative to its current position

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the 
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        relative_to_tcp : bool, optional
            if True, consider the current position relative to the end-effector frame.
            if False, consider the current position relative to the robot base frame.
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool or str
            True if the trajectory is succesfully executed.
            If the IK solver fails, return "ik_not_found"
        """
        new_pose = transformations.transform_pose(self.end_effector(), transformation, rotated_frame=relative_to_tcp)
        return self.set_target_pose(pose=new_pose, target_time=target_time, wait=wait)

### FT sensor control ###

    def zero_ft_sensor(self):
        """ 
        Reset force-torque sensor readings to zeros.
        """
        if not rospy.has_param("use_gazebo_sim"):
            # First try to zero FT from ur_driver
            self._zero_ft()
        # Then update filtered one
        self._zero_ft_filtered()

    def set_ft_filtering(self, active=True):
        """ 
        Enable/disable a low-pass filter.
        If active, the readings returned from self.get_wrench will have been filtered.
        otherwise, the raw data from the sensor's topic will be returned.

        The filtering is done in an external topic. See scripts/ft_filter.py
        """
        self._ft_filtered(active)
