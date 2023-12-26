# Gripper action
import actionlib
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

from ur_control import utils
# Link attacher
try:
    from gazebo_ros_link_attacher.srv import Attach, AttachRequest
except ImportError:
    print("Grasping pluging can't be loaded")

try:
    import robotiq_msgs.msg
except ImportError:
    print("Robotiq gripper can't be load. robotiq_msgs required.")

class GripperController(object):
    def __init__(self, namespace='', prefix=None, timeout=5.0, attach_link='robot::wrist_3_link'):
        self.ns = utils.solve_namespace(namespace)
        self.prefix = prefix if prefix is not None else ''
        node_name = "gripper_controller"
        self.gripper_type = rospy.get_param(self.ns + node_name + "/gripper_type", None)
        if self.gripper_type is None:
            rospy.logwarn("gripper_type was not define. Using configuration for Robotiq 85")
            self.gripper_type = "85"
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


class RobotiqGripper():
    def __init__(self, namespace="", timeout=2):
        self.ns = namespace

        self.opening_width = 0.0

        self.sub_gripper_status_ = rospy.Subscriber(self.ns + "/gripper_status", robotiq_msgs.msg.CModelCommandFeedback, self._gripper_status_callback)
        self.gripper = actionlib.SimpleActionClient(self.ns + '/gripper_action_controller', robotiq_msgs.msg.CModelCommandAction)
        success = self.gripper.wait_for_server(rospy.Duration(timeout))
        if success:
            rospy.loginfo("=== Connected to ROBOTIQ gripper ===")
        else:
            rospy.logerr("Unable to connect to ROBOTIQ gripper: %s" % (self.ns + "/gripper_status"))

    def _gripper_status_callback(self, msg):
        self.opening_width = msg.position  # [m]

    def get_position(self):
        return self.opening_width

    def close(self, force=40.0, velocity=1.0, wait=True):
        return self.send_command("close", force=force, velocity=velocity, wait=wait)

    def open(self, velocity=1.0, wait=True, opening_width=None):
        command = opening_width if opening_width else "open"
        return self.send_command(command, wait=wait, velocity=velocity)

    def send_command(self, command, force=40.0, velocity=1.0, wait=True):
        """
        command: "open", "close" or opening width
        force: Gripper force in N. From 40 to 100
        velocity: Gripper speed. From 0.013 to 0.1
        attached_last_object: bool, Attach/detach last attached object if set to True

        Use a slow closing speed when using a low gripper force, or the force might be unexpectedly high.
        """
        goal = robotiq_msgs.msg.CModelCommandGoal()
        goal.velocity = velocity
        goal.force = force
        if command == "close":
            goal.position = 0.0
        elif command == "open":
            goal.position = 0.140
        else:
            goal.position = command     # This sets the opening width directly

        self.gripper.send_goal(goal)
        rospy.logdebug("Sending command " + str(command) + " to gripper: " + self.ns)
        if wait:
            self.gripper.wait_for_result(rospy.Duration(5.0))  # Default wait time: 5 s
            result = self.gripper.get_result()
            return True if result else False
        else:
            return True
