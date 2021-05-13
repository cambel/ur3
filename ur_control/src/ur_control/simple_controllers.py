#!/usr/bin/env python
import actionlib
import copy
import collections
import rospy
from ur_control import utils, filters, conversions
import os
import numpy as np
from std_msgs.msg import Float64

# Joint trajectory action
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped


class JointControllerBase(object):
  """
  Base class for the Joint Position Controllers. It subscribes to the C{joint_states} topic by default.
  """
  def __init__(self, namespace='/', timeout=5):
    """
    JointControllerBase constructor. It subscribes to the C{joint_states} topic and informs after 
    successfully reading a message from the topic.
    @type namespace: string
    @param namespace: Override ROS namespace manually. Useful when controlling several robots 
    @type  timeout: float
    @param timeout: Time in seconds that will wait for the controller
    from the same node.
    """
    self.ns = utils.solve_namespace(namespace)
    self._jnt_positions_hist = collections.deque(maxlen=24)
    # Set-up publishers/subscribers
    self._js_sub = rospy.Subscriber('%sjoint_states' % self.ns, JointState, self.joint_states_cb, queue_size=1)
    rospy.logdebug('Waiting for [%sjoint_states] topic' % self.ns)
    start_time = rospy.get_time()
    while not hasattr(self, '_joint_names'):
      if (rospy.get_time() - start_time) > timeout:
        rospy.logerr('Timed out waiting for joint_states topic')
        return
      rospy.sleep(0.01)
      if rospy.is_shutdown():
        return
    self.rate = utils.read_parameter('{0}joint_state_controller/publish_rate'.format(self.ns), 500)
    self._rate = rospy.Rate(self.rate)
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
    valid_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 
                         'elbow_joint', 'wrist_1_joint',
                         'wrist_2_joint', 'wrist_3_joint',
                         'hande_left_finger_joint',
                         'hande_right_finger_joint',]
    position = []
    velocity = []
    effort = []
    name = []
    for joint_name in valid_joint_names:
      if joint_name in msg.name:
        idx = msg.name.index(joint_name)
        name.append(msg.name[idx])
        effort.append(msg.effort[idx])
        velocity.append(msg.velocity[idx])
        position.append(msg.position[idx])
    if set(name) == set(valid_joint_names):
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
  def __init__(self, namespace='', timeout=5.0):
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
    super(JointPositionController, self).__init__(namespace, timeout=timeout)
    if not hasattr(self, '_joint_names'):
      raise rospy.ROSException('JointPositionController timed out waiting joint_states topic: {0}'.format(namespace))

    self._cmd_pub = rospy.Publisher("/joint_command", JointState, queue_size=3)
    self.joint_state = JointState()
    self.joint_state.position = np.array([0.0] * self._num_joints)
    self.joint_state.velocity = [0.0] * self._num_joints
    self.joint_state.effort = [0.0] * self._num_joints

    # for Franka Panda Robot
    self.joint_state.name = [
    'shoulder_pan_joint', 
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
    'hande_left_finger_joint',
    'hande_right_finger_joint',
    ]
    rospy.loginfo('JointPositionController initialized. ns: {0}'.format(namespace))
  
  def set_joint_positions(self, jnt_positions, wait=True):
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
    self.joint_state.position = jnt_positions
    self._cmd_pub.publish(self.joint_state)
    if wait:
      self._rate.sleep()
  
  def valid_jnt_command(self, command):
    """
    It validates that the length of a joint command is equal to the number of joints
    @type command: list
    @param command: Joint command to be validated
    @rtype: bool
    @return: True if the joint command is valid
    """
    return ( len(command) == self._num_joints )
