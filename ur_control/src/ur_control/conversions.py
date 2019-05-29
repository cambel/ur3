#! /usr/bin/env python
import rospy
import PyKDL
import numpy as np
import tf.transformations as tr
# Messages
from geometry_msgs.msg import (Point, Quaternion, Pose, Vector3, Transform,
                                                                        Wrench)
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# OpenRAVE types <--> Numpy types
def from_dict(transform_dict):
  """
  Converts a dictionary with the fields C{rotation} and C{translation}
  into a homogeneous transformation of type C{np.array}.
  @type transform_dict:  dict
  @param transform_dict: The dictionary to be converted.
  @rtype: np.array
  @return: The resulting numpy array
  """
  T = tr.quaternion_matrix(np.array(transform_dict['rotation']))
  T[:3,3] = np.array(transform_dict['translation'])
  return T

# PyKDL types <--> Numpy types
def from_kdl_vector(vector):
  """
  Converts a C{PyKDL.Vector} with fields into a numpy array.
  @type  vector: PyKDL.Vector
  @param vector: The C{PyKDL.Vector} to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  return np.array([vector.x(),vector.y(),vector.z()])

def from_kdl_twist(twist):
  """
  Converts a C{PyKDL.Twist} with fields into a numpy array.
  @type  twist: PyKDL.Twist
  @param twist: The C{PyKDL.Twist} to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  array = np.zeros(6)
  array[:3] = from_kdl_vector(twist.vel)
  array[3:] = from_kdl_vector(twist.rot)
  return array


# ROS types <--> Numpy types
def from_point(msg):
  """
  Converts a C{geometry_msgs/Point} ROS message into a numpy array.
  @type  msg: geometry_msgs/Point
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  return from_vector3(msg)

def from_pose(msg):
  """
  Converts a C{geometry_msgs/Pose} ROS message into a numpy array (Homogeneous transformation 4x4).
  @type  msg: geometry_msgs/Pose
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  T = tr.quaternion_matrix(from_quaternion(msg.orientation))
  T[:3,3] = from_point(msg.position)
  return T

def from_quaternion(msg):
  """
  Converts a C{geometry_msgs/Quaternion} ROS message into a numpy array.
  @type  msg: geometry_msgs/Quaternion
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  return np.array([msg.x, msg.y, msg.z, msg.w])

def from_roi(msg):
  top_left = np.array([msg.x_offset, msg.y_offset])
  bottom_right = top_left + np.array([msg.width, msg.height])
  return [top_left, bottom_right]

def from_transform(msg):
  """
  Converts a C{geometry_msgs/Transform} ROS message into a numpy array.
  @type  msg: geometry_msgs/Transform
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  T = tr.quaternion_matrix(from_quaternion(msg.rotation))
  T[:3,3] = from_vector3(msg.translation)
  return T

def from_vector3(msg):
  """
  Converts a C{geometry_msgs/Vector3} ROS message into a numpy array.
  @type  msg: geometry_msgs/Vector3
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  return np.array([msg.x, msg.y, msg.z])

def from_wrench(msg):
  """
  Converts a C{geometry_msgs/Wrench} ROS message into a numpy array.
  @type  msg: geometry_msgs/Wrench
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
  array = np.zeros(6)
  array[:3] = from_vector3(msg.force)
  array[3:] = from_vector3(msg.torque)
  return array

def to_quaternion(array):
  """
  Converts a numpy array into a C{geometry_msgs/Quaternion} ROS message.
  @type  array: np.array
  @param array: The position as numpy array
  @rtype: geometry_msgs/Quaternion
  @return: The resulting ROS message
  """
  return Quaternion(*array)

def to_point(array):
  """
  Converts a numpy array into a C{geometry_msgs/Point} ROS message.
  @type  array: np.array
  @param array: The position as numpy array
  @rtype: geometry_msgs/Point
  @return: The resulting ROS message
  """
  return Point(*array)


def to_pose(T):
  """
  Converts a homogeneous transformation (4x4) into a C{geometry_msgs/Pose} ROS message.
  @type  T: np.array
  @param T: The homogeneous transformation
  @rtype: geometry_msgs/Pose
  @return: The resulting ROS message
  """
  pos = Point(*T[:3,3])
  quat = Quaternion(*tr.quaternion_from_matrix(T))
  return Pose(pos, quat)

def to_roi(top_left, bottom_right):
  msg = RegionOfInterest()
  msg.x_offset = round(top_left[0])
  msg.y_offset = round(top_left[1])
  msg.width = round(abs(bottom_right[0]-top_left[0]))
  msg.height = round(abs(bottom_right[1]-top_left[1]))
  return msg

def to_transform(T):
  """
  Converts a homogeneous transformation (4x4) into a C{geometry_msgs/Transform}
  ROS message.
  @type  T: np.array
  @param T: The homogeneous transformation
  @rtype: geometry_msgs/Transform
  @return: The resulting ROS message
  """
  translation = Vector3(*T[:3,3])
  rotation = Quaternion(*tr.quaternion_from_matrix(T))
  return Transform(translation, rotation)

def to_vector3(array):
  """
  Converts a numpy array into a C{geometry_msgs/Vector3} ROS message.
  @type  array: np.array
  @param array: The vector as numpy array
  @rtype: geometry_msgs/Vector3
  @return: The resulting ROS message
  """
  return Vector3(*array)


def to_wrench(array):
  """
  Converts a numpy array into a C{geometry_msgs/Wrench} ROS message.
  @type  array: np.array
  @param array: The wrench as numpy array
  @rtype: geometry_msgs/Wrench
  @return: The resulting ROS message
  """
  msg = Wrench()
  msg.force = to_vector3(array[:3])
  msg.torque = to_vector3(array[3:])
  return msg


# RViz types <--> Numpy types
def from_rviz_vector(value, dtype=float):
  """
  Converts a RViz property vector in the form C{X;Y;Z} into a numpy array.
  @type  value: str
  @param value: The RViz property vector
  @type  dtype: type
  @param dtype: The type of mapping to be done. Typically C{float} or C{int}.
  @rtype: array
  @return: The resulting numpy array
  """
  strlst = value.split(';')
  return np.array( map(dtype, strlst) )
