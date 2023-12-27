#!/usr/bin/python
#
# Send joint values to UR5 using messages
#

from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState # Used for subscribing to the UR.
import rospy

def _observation_callback(message):
    """This callback is set on the subscriber node in self.__init__().
    It's called by ROS every 40 ms while the subscriber is listening.
    Primarily updates the present and latest times.
    This callback is invoked asynchronously, so is effectively a
    "subscriber thread", separate from the control flow of the rest of
    GPS, which runs in the "main thread".
    message: observation from the robot to store each listen."""

    print(message)

def main():

    rospy.init_node('send_joints')
    pub = rospy.Publisher('/position_trajectory_controller/command',
                          JointTrajectory,
                          queue_size=10)
    sub = rospy.Subscriber('/position_trajectory_controller/state',
                           JointTrajectoryControllerState,
                           _observation_callback, queue_size=10)
    # Create the topic message
    traj = JointTrajectory()
    traj.header = Header()
    # Joint names for UR5
    traj.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                        'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                        'wrist_3_joint']

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        traj.header.stamp = rospy.Time.now()
        pts = JointTrajectoryPoint()
        pts.positions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        pts.time_from_start = rospy.Duration(0.1)

        # Set the points to the trajectory
        traj.points = []
        traj.points.append(pts)
        # Publish the message
        pub.publish(traj)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
