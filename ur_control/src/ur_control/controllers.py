#!/usr/bin/env python
import actionlib
import copy
import collections
import rospy
from ur_control import utils, filters, conversions, constants
import numpy as np
from std_msgs.msg import Float64
from controller_manager_msgs.srv import ListControllers
# Joint trajectory action
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import WrenchStamped
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# Gripper action
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
# Link attacher
try:
    from gazebo_ros_link_attacher.srv import Attach, AttachRequest
except ImportError:
    print("Grasping pluging can't be loaded")

class GripperController(object):
    def __init__(self, namespace='', prefix=None, timeout=5.0, attach_link='robot::wrist_3_link'):
        self.ns = utils.solve_namespace(namespace)
        self.prefix = prefix if prefix is not None else ''
        node_name = "gripper_controller"
        self.gripper_type = str(rospy.get_param(self.ns + node_name + "/gripper_type"))
        self.valid_joint_names = []
        if rospy.has_param(self.ns + node_name + "/joint"):
            self.valid_joint_names = [rospy.get_param(self.ns + node_name + "/joint")]
        elif rospy.has_param(self.ns + node_name + "/joint"):
            self.valid_joint_names = rospy.get_param(self.ns + node_name + "/joints")
        else:
            rospy.logerr("Couldn't find valid joints params in %s" % (self.ns + node_name))
            return

        if self.gripper_type == "hand-e":
            self._max_gap = 0.025 * 2.0
            self._to_open = 0.0
            self._to_close = self._max_gap
        elif self.gripper_type == "85":
            self._max_gap = 0.085
            self._to_open = self._max_gap
            self._to_close = 0.001
            self._max_angle = 0.8028
        elif self.gripper_type == "140":
            self._max_gap = 0.140
            self._to_open = self._max_gap
            self._to_close = 0.001
            self._max_angle = 0.69
        self._js_sub = rospy.Subscriber('joint_states', JointState, self.joint_states_cb, queue_size=1)

        retry = False
        rospy.logdebug('Waiting for [%sjoint_states] topic' % self.ns)
        start_time = rospy.get_time()
        while not hasattr(self, '_joint_names'):
            if (rospy.get_time() - start_time) > timeout and not retry:
                # Re-try with namespace
                self._js_sub = rospy.Subscriber('%sjoint_states' % self.ns, JointState, self.joint_states_cb, queue_size=1)
                start_time = rospy.get_time()
                retry = True
                continue
            elif (rospy.get_time() - start_time) > timeout and retry:
                rospy.logerr('Timed out waiting for joint_states topic')
                return
            rospy.sleep(0.01)
            if rospy.is_shutdown():
                return

        attach_plugin = rospy.get_param("grasp_plugin", default=False)
        if attach_plugin:
            try:
                # gazebo_ros link attacher
                self.attach_link = attach_link
                self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
                self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
                rospy.logdebug('Waiting for service: {0}'.format(self.attach_srv.resolved_name))
                rospy.logdebug('Waiting for service: {0}'.format(self.detach_srv.resolved_name))
                self.attach_srv.wait_for_service()
                self.detach_srv.wait_for_service()
            except Exception:
                rospy.logerr("Fail to load grasp plugin services. Make sure to launch the right Gazebo world!")
        # Gripper action server
        action_server = self.ns + node_name + '/gripper_cmd'
        self._client = actionlib.SimpleActionClient(action_server, GripperCommandAction)
        self._goal = GripperCommandGoal()
        rospy.logdebug('Waiting for [%s] action server' % action_server)
        server_up = self._client.wait_for_server(timeout=rospy.Duration(timeout))
        if not server_up:
            rospy.logerr('Timed out waiting for Gripper Command'
                         ' Action Server to connect. Start the action server'
                         ' before running this node.')
            raise rospy.ROSException('GripperCommandAction timed out: {0}'.format(action_server))
        rospy.logdebug('Successfully connected to [%s]' % action_server)
        rospy.loginfo('GripperCommandAction initialized. ns: {0}'.format(self.ns))

    def close(self, wait=True):
        return self.command(0.0, percentage=True, wait=wait)

    def command(self, value, percentage=False, wait=True):
        """ assume command given in percentage otherwise meters 
            percentage bool: If True value value assumed to be from 0.0 to 1.0
                                     where 1.0 is open and 0.0 is close
                             If False value value assume to be from 0.0 to max_gap
        """
        if value == "close":
            return self.close()
        elif value == "open":
            return self.open()

        if self.gripper_type == "85" or self.gripper_type == "140":
            if percentage:
                value = np.clip(value, 0.0, 1.0)
                cmd = (value) * self._max_gap
            else:
                cmd = np.clip(value, 0.0, self._max_gap)
                cmd = (value)
            angle = self._distance_to_angle(cmd)
            self._goal.command.position = angle
        if self.gripper_type == "hand-e":
            cmd = 0.0
            if percentage:
                value = np.clip(value, 0.0, 1.0)
                cmd = (1.0 - value) * self._max_gap / 2.0
            else:
                cmd = np.clip(value, 0.0, self._max_gap)
                cmd = (self._max_gap - value) / 2.0
            self._goal.command.position = cmd
        if wait:
            self._client.send_goal_and_wait(self._goal, execute_timeout=rospy.Duration(2))
        else:
            self._client.send_goal(self._goal)
        rospy.sleep(0.05)
        return True

    def _distance_to_angle(self, distance):
        distance = np.clip(distance, 0, self._max_gap)
        angle = (self._max_gap - distance) * self._max_angle / self._max_gap
        return angle

    def _angle_to_distance(self, angle):
        angle = np.clip(angle, 0, self._max_angle)
        distance = (self._max_angle - angle) * self._max_gap / self._max_angle
        return distance

    def get_result(self):
        return self._client.get_result()

    def get_state(self):
        return self._client.get_state()

    def grab(self, link_name):
        parent = self.attach_link.split('::')
        child = link_name.split('::')
        req = AttachRequest()
        req.model_name_1 = parent[0]
        req.link_name_1 = parent[1]
        req.model_name_2 = child[0]
        req.link_name_2 = child[1]
        res = self.attach_srv.call(req)
        return res.ok

    def open(self, wait=True):
        return self.command(1.0, percentage=True, wait=wait)

    def release(self, link_name):
        parent = self.attach_link.rsplit('::')
        child = link_name.rsplit('::')
        req = AttachRequest()
        req.model_name_1 = parent[0]
        req.link_name_1 = parent[1]
        req.model_name_2 = child[0]
        req.link_name_2 = child[1]
        res = self.detach_srv.call(req)
        return res.ok

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        return self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def get_position(self):
        """
        Returns the current joint positions of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint positions of the UR robot.
        """
        if self.gripper_type == "hand-e":
            return self._max_gap - (self._current_jnt_positions[0] * 2.0)
        else:
            return self._angle_to_distance(self._current_jnt_positions[0])

    def joint_states_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{joint_states} topic.
        @type  msg: sensor_msgs/JointState
        @param msg: The JointState message published by the RT hardware interface.
        """

        position = []
        velocity = []
        effort = []
        name = []

        for joint_name in self.valid_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                name.append(msg.name[idx])
                effort.append(msg.effort[idx])
                velocity.append(msg.velocity[idx])
                position.append(msg.position[idx])

        if set(name) == set(self.valid_joint_names):
            self._current_jnt_positions = np.array(position)
            self._current_jnt_velocities = np.array(velocity)
            self._current_jnt_efforts = np.array(effort)
            self._joint_names = list(name)


class JointControllerBase(object):
    """
    Base class for the Joint Position Controllers. It subscribes to the C{joint_states} topic by default.
    """

    def __init__(self, namespace, timeout, joint_names=None):
        """
        JointControllerBase constructor. It subscribes to the C{joint_states} topic and informs after 
        successfully reading a message from the topic.
        @type namespace: string
        @param namespace: Override ROS namespace manually. Useful when controlling several robots 
        @type  timeout: float
        @param timeout: Time in seconds that will wait for the controller
        from the same node.
        """
        self.valid_joint_names = constants.JOINT_ORDER if joint_names is None else joint_names

        self.ns = utils.solve_namespace(namespace)
        self._jnt_positions_hist = collections.deque(maxlen=24)
        # Set-up publishers/subscribers
        self._js_sub = rospy.Subscriber('joint_states', JointState, self.joint_states_cb, queue_size=1)
        retry = False
        rospy.logdebug('Waiting for [%sjoint_states] topic' % self.ns)
        start_time = rospy.get_time()
        while not hasattr(self, '_joint_names'):
            if (rospy.get_time() - start_time) > timeout and not retry:
                # Re-try with namespace
                self._js_sub = rospy.Subscriber('%sjoint_states' % self.ns, JointState, self.joint_states_cb, queue_size=1)
                start_time = rospy.get_time()
                retry = True
                continue
            elif (rospy.get_time() - start_time) > timeout and retry:
                rospy.logerr('Timed out waiting for joint_states topic')
                return
            rospy.sleep(0.01)
            if rospy.is_shutdown():
                return
        self.rate = utils.read_parameter('{0}joint_state_controller/publish_rate'.format(self.ns), 500)
        self._num_joints = len(self._joint_names)
        rospy.logdebug('Topic [%sjoint_states] found' % self.ns)

    def disconnect(self):
        """
        Disconnects from the joint_states topic. Useful to ligthen the use of system resources.
        """
        self._js_sub.unregister()

    def get_joint_efforts(self):
        """
        Returns the current joint efforts of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint efforts of the UR robot.
        """
        return np.array(self._current_jnt_efforts)

    def get_joint_positions(self):
        """
        Returns the current joint positions of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint positions of the UR robot.
        """
        return np.array(self._current_jnt_positions)

    def get_joint_positions_hist(self):
        """
        Returns the current joint positions of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint positions of the UR robot.
        """
        return list(self._jnt_positions_hist)

    def get_joint_velocities(self):
        """
        Returns the current joint velocities of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint velocities of the UR robot.
        """
        return np.array(self._current_jnt_velocities)

    def joint_states_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{joint_states} topic.
        @type  msg: sensor_msgs/JointState
        @param msg: The JointState message published by the RT hardware interface.
        """
        position = []
        velocity = []
        effort = []
        name = []
        for joint_name in self.valid_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                name.append(msg.name[idx])
                if msg.effort:
                    effort.append(msg.effort[idx])
                if msg.velocity:
                    velocity.append(msg.velocity[idx])
                position.append(msg.position[idx])
        if set(name) == set(self.valid_joint_names):
            self._current_jnt_positions = np.array(position)
            self._jnt_positions_hist.append(self._current_jnt_positions)
            self._current_jnt_velocities = np.array(velocity)
            self._current_jnt_efforts = np.array(effort)
            self._joint_names = list(name)


class JointPositionController(JointControllerBase):
    """
    Interface class to control the UR robot using a Joint Position Control approach. 
    If you C{set_joint_positions} to a value very far away from the current robot position, 
    it will move at its maximum speed/acceleration and will even move the base of the robot, so, B{use with caution}.
    """

    def __init__(self, namespace='', timeout=5.0, joint_names=None):
        """
        C{JointPositionController} constructor. It creates the required publishers for controlling 
        the UR robot. Given that it inherits from C{JointControllerBase} it subscribes 
        to C{joint_states} by default.
        @type namespace: string
        @param namespace: Override ROS namespace manually. Useful when controlling several robots 
        from the same node.
        @type  timeout: float
        @param timeout: Time in seconds that will wait for the controller
        """
        super(JointPositionController, self).__init__(namespace, timeout=timeout, joint_names=joint_names)
        if not hasattr(self, '_joint_names'):
            raise rospy.ROSException('JointPositionController timed out waiting joint_states topic: {0}'.format(namespace))
        self._cmd_pub = dict()
        for joint in self._joint_names:
            print(('%s%s/command' % (self.ns, joint)))
            self._cmd_pub[joint] = rospy.Publisher('%s%s/command' % (self.ns, joint), Float64, queue_size=3)
        # Wait for the joint position controllers
        controller_list_srv = self.ns + 'controller_manager/list_controllers'
        rospy.logdebug('Waiting for the joint position controllers...')
        rospy.wait_for_service(controller_list_srv, timeout=timeout)
        list_controllers = rospy.ServiceProxy(controller_list_srv, ListControllers)
        expected_controllers = (joint_names if joint_names is not None else constants.JOINT_ORDER)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (rospy.get_time() - start_time) > timeout:
                raise rospy.ROSException('JointPositionController timed out waiting for the controller_manager: {0}'.format(namespace))
            rospy.sleep(0.01)
            found = 0
            try:
                res = list_controllers()
                for state in res.controller:
                    if state.name in expected_controllers:
                        found += 1
            except:
                pass
            if found == len(expected_controllers):
                break
        rospy.loginfo('JointPositionController initialized. ns: {0}'.format(namespace))

    def set_joint_positions(self, jnt_positions):
        """
        Sets the joint positions of the robot. The values are send directly to the robot. If the goal 
        is too far away from the current robot position, it will move at its maximum speed/acceleration 
        and will even move the base of the robot, so, B{use with caution}.
        @type jnt_positions: list
        @param jnt_positions: Joint positions command.
        """
        if not self.valid_jnt_command(jnt_positions):
            rospy.logwarn('A valid joint positions command should have %d elements' % (self._num_joints))
            return
        # Publish the point for each joint
        for name, q in zip(self._joint_names, jnt_positions):
            try:
                self._cmd_pub[name].publish(q)
            except:
                print("some error")
                pass

    def valid_jnt_command(self, command):
        """
        It validates that the length of a joint command is equal to the number of joints
        @type command: list
        @param command: Joint command to be validated
        @rtype: bool
        @return: True if the joint command is valid
        """
        return (len(command) == self._num_joints)


class JointTrajectoryController(JointControllerBase):
    """
    This class creates a C{SimpleActionClient} that connects to the 
    C{trajectory_controller/follow_joint_trajectory} action server. Using this 
    interface you can control the robot by adding points to the trajectory. Each point 
    requires the goal position and the goal time. The velocity and acceleration are optional.

    The controller is templated to work with multiple trajectory representations. By default, 
    a B{linear} interpolator is provided, but it's possible to support other representations. 
    The interpolator uses the following interpolation strategies depending on the waypoint 
    specification:

      - B{Linear}: Only position is specified. Guarantees continuity at the position level. 
        B{Discouraged} because it yields trajectories with discontinuous velocities at the waypoints.
      - B{Cubic}: Position and velocity are specified. Guarantees continuity at the velocity level.
      - B{Quintic}: Position, velocity and acceleration are specified: Guarantees continuity at 
        the acceleration level.
    """

    def __init__(self, publisher_name='arm_controller', namespace='', timeout=5.0, joint_names=None):
        """
        JointTrajectoryController constructor. It creates the C{SimpleActionClient}.
        @type namespace: string
        @param namespace: Override ROS namespace manually. Useful when controlling several 
        robots from the same node.
        @type  timeout: float
        @param timeout: Time in seconds that will wait for the controller
        """
        super(JointTrajectoryController, self).__init__(namespace, timeout=timeout, joint_names=joint_names)
        action_server = self.ns + publisher_name + '/follow_joint_trajectory'
        self._client = actionlib.SimpleActionClient(action_server, FollowJointTrajectoryAction)
        self._goal = FollowJointTrajectoryGoal()
        rospy.logdebug('Waiting for [%s] action server' % action_server)
        topics = [item for sublist in rospy.get_published_topics() for item in sublist]
        if action_server+"/goal" not in topics:
            raise rospy.ROSException("Action server not found")
        server_up = self._client.wait_for_server(timeout=rospy.Duration(timeout))
        if not server_up:
            rospy.logerr('Timed out waiting for Joint Trajectory'
                         ' Action Server to connect. Start the action server'
                         ' before running this node.')
            raise rospy.ROSException('JointTrajectoryController timed out: {0}'.format(action_server))
        rospy.logdebug('Successfully connected to [%s]' % action_server)
        # Get a copy of joint_names
        if not hasattr(self, '_joint_names'):
            raise rospy.ROSException('JointTrajectoryController timed out waiting joint_states topic: {0}'.format(self.ns))
        self._goal.trajectory.joint_names = copy.deepcopy(self._joint_names)
        rospy.loginfo('JointTrajectoryController initialized. ns: {0}'.format(self.ns))

    def add_point(self, positions, time, velocities=None, accelerations=None):
        """
        Adds a point to the trajectory. Each point must be specified by the goal position and 
        the goal time. The velocity and acceleration are optional.
        @type  positions: list
        @param positions: The goal position in the joint space
        @type  time: float
        @param time: The time B{from start} when the robot should arrive at the goal position.
        @type  velocities: list
        @param velocities: The velocity of arrival at the goal position. If not given zero 
        velocity is assumed.
        @type  accelerations: list
        @param accelerations: The acceleration of arrival at the goal position. If not given 
        zero acceleration is assumed.
        """
        point = JointTrajectoryPoint()
        point.positions = copy.deepcopy(positions)
        if type(velocities) == type(None):
            point.velocities = [0] * self._num_joints
        else:
            point.velocities = copy.deepcopy(velocities)
        if type(accelerations) == type(None):
            point.accelerations = [0] * self._num_joints
        else:
            point.accelerations = copy.deepcopy(accelerations)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def clear_points(self):
        """
        Clear all points in the trajectory.
        """
        self._goal.trajectory.points = []

    def get_num_points(self):
        """
        Returns the number of points currently added to the trajectory
        @rtype: int
        @return: Number of points currently added to the trajectory
        """
        return len(self._goal.trajectory.points)

    def get_result(self):
        """
        Returns the result B{after} the execution of the trajectory. Possible values:
          - FollowJointTrajectoryResult.SUCCESSFUL = 0
          - FollowJointTrajectoryResult.INVALID_GOAL = -1
          - FollowJointTrajectoryResult.INVALID_JOINTS = -2
          - FollowJointTrajectoryResult.OLD_HEADER_TIMESTAMP = -3
          - FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED = -4
          - FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED = -5
        @rtype: int
        @return: result B{after} the execution of the trajectory
        """
        return self._client.get_result()

    def get_state(self):
        """
        Returns the status B{during} the execution of the trajectory. Possible values:
          - GoalStatus.PENDING=0
          - GoalStatus.ACTIVE=1
          - GoalStatus.PREEMPTED=2
          - GoalStatus.SUCCEEDED=3
          - GoalStatus.ABORTED=4
          - GoalStatus.REJECTED=5
          - GoalStatus.PREEMPTING=6
          - GoalStatus.RECALLING=7
          - GoalStatus.RECALLED=8
          - GoalStatus.LOST=9
        @rtype: int
        @return: result B{after} the execution of the trajectory
        """
        return self._client.get_state()

    def set_trajectory(self, trajectory):
        """
        Sets the goal trajectory directly. B{It only copies} the C{trajectory.points} field. 
        @type  trajectory: trajectory_msgs/JointTrajectory
        @param trajectory: The goal trajectory
        """
        self._goal.trajectory.points = copy.deepcopy(trajectory.points)

    def start(self, delay=0.1, wait=False):
        """
        Starts the trajectory. It sends the C{FollowJointTrajectoryGoal} to the action server.
        @type  delay: float
        @param delay: Delay (in seconds) before executing the trajectory
        """
        num_points = len(self._goal.trajectory.points)
        rospy.logdebug('Executing Joint Trajectory with {0} points'.format(num_points))
        self._goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(delay)
        if wait:
            self._client.send_goal_and_wait(self._goal)
        else:
            self._client.send_goal(self._goal)

    def stop(self):
        """
        Stops an active trajectory. If there is not active trajectory an error will be shown in the console.
        """
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        """
        Waits synchronously (with a timeout) until the trajectory action server gives a result.
        @type  timeout: float
        @param timeout: The amount of time we will wait
        @rtype: bool
        @return: True if the server connected in the allocated time. False on timeout
        """
        return self._client.wait_for_result(timeout=rospy.Duration(timeout))


class FTsensor(object):

    def __init__(self, namespace='', timeout=3.0):
        ns = utils.solve_namespace(namespace)
        self.raw_msg = None
        self.rate = 500
        self.wrench_rate = 500
        self.wrench_filter = filters.ButterLowPass(2.5, self.rate, 2)
        self.wrench_window = int(100)
        assert(self.wrench_window >= 5)
        self.wrench_queue = collections.deque(maxlen=self.wrench_window)
        rospy.Subscriber('%s' % ns, WrenchStamped, self.cb_raw)
        if not utils.wait_for(lambda: self.raw_msg is not None, timeout=timeout):
            rospy.logerr('Timed out waiting for {0} topic'.format(ns))
            return
        rospy.loginfo('FTSensor successfully initialized')
        rospy.sleep(1.0)

    def add_wrench_observation(self, wrench):
        self.wrench_queue.append(np.array(wrench))

    def cb_raw(self, msg):
        self.raw_msg = copy.deepcopy(msg)
        self.add_wrench_observation(conversions.from_wrench(self.raw_msg.wrench))

    # function to filter out high frequency signal
    def get_filtered_wrench(self, hist_size=1):
        if len(self.wrench_queue) < self.wrench_window:
            return None
        wrench_filtered = self.wrench_filter(np.array(self.wrench_queue))
        if hist_size == 1:
            return wrench_filtered[-hist_size, :]
        else:
            return wrench_filtered[-hist_size:, :]
