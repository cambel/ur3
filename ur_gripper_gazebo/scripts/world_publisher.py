#!/usr/bin/env python3
# tf2 workaround for Python3
import sys
sys.path[:0] = ['/usr/local/lib/python3.6/dist-packages/'] 

import tf
import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates, ModelState
from apriltag_ros.msg import AprilTagDetectionArray
from tf2_msgs.msg import TFMessage
from ur_control import conversions, transformations
from pyquaternion import Quaternion

rospy.init_node('gazebo_models_tf')

def gazebo_callback(msg):
    global robot
    robot = []
    world = np.zeros(7)
    world[6] = 1
    for i, obj_name in enumerate(msg.name):
        if obj_name.endswith("robot"):
            pose = msg.pose[i]
            robot = np.concatenate([conversions.from_point(pose.position), conversions.from_quaternion(pose.orientation)])
            # subtract robot_position to camera_position
            world[:3] -= robot[:3]
            world[:3] = Quaternion(np.roll(robot[3:],1)).inverse.rotate(world[:3])
            world[3:] = np.roll((Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90).inverse * Quaternion(np.roll(world[3:],1))).elements,-1)
            world[3:] = transformations.vector_from_pyquaternion(transformations.vector_to_pyquaternion([0.5,-0.5,-0.5,0.5]).inverse)
            # world[3:] = np.roll((Quaternion(axis=[0.0, 0.0, 0.0], degrees=0).inverse * Quaternion(np.roll(world[3:],1))).elements,-1)
            # print(world)orientation
            br.sendTransform(world[:3],
                            world[3:],
                            rospy.Time.now(),
                            "sim_world",
                            "base_link")

rospy.Subscriber("/gazebo/model_states", ModelStates, gazebo_callback)

br = tf.TransformBroadcaster()
rate = rospy.Rate(100.0)
while not rospy.is_shutdown():
    rate.sleep()