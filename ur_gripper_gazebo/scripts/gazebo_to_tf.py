#!/usr/bin/python
import rospy

from gazebo_msgs.msg import ModelStates

import tf
import rospy

from ur_control import conversions


class GazeboToTf:
    """ Class to handle ROS-Gazebo model respawn """

    def __init__(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)
        self.tf_publisher = tf.TransformBroadcaster()
        self.time_now = rospy.Time.now()

    def callback(self, data):
        if rospy.Time.now() is not self.time_now:
            for i in range(len(data.name)):
                # get model state of all objects
                self.tf_publisher.sendTransform(conversions.from_point(data.pose[i].position),
                                                conversions.from_quaternion(data.pose[i].orientation),
                                                rospy.Time.now(),
                                                data.name[i],
                                                "world")
            self.time_now = rospy.Time.now()


if __name__ == '__main__':
    rospy.init_node('gazebo_to_tf')
    g2tf = GazeboToTf()
    rospy.spin()
