import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class Mouse6D():
    """ Subscribe to the 3DConnextion mouse convert messages """
    def __init__(self):
        
        twist_sub = rospy.Subscriber('spacenav/twist', Twist, callback=self.twist_cb, queue_size=1)
        joy_sub = rospy.Subscriber('spacenav/joy', Joy, callback=self.joy_cb, queue_size=1)

        self.twist = None
        self.joy_axes = None
        self.joy_buttons = None
        
        # Wait for publisher
        rospy.sleep(0.01)

    def twist_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{spacenav/twist} topic.
        @type  msg: geometry_msgs/JointState
        @param msg: The Twist message published by the 3DConnextion hardware interface.
        """ 
        self.twist = []
        self.twist.append(msg.linear.x)
        self.twist.append(msg.linear.y)
        self.twist.append(msg.linear.z)
        self.twist.append(msg.angular.x)
        self.twist.append(msg.angular.y)
        self.twist.append(msg.angular.z)

    def joy_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{spacenav/joy} topic.
        @type  msg: sensor_msgs/Joy
        @param msg: The Joy message published by the 3DConnextion hardware interface.
        """
        self.joy_axes = msg.axes
        self.joy_buttons = msg.buttons

def main():
    rospy.init_node("Mouse6D")
    Mouse6D()
    rospy.sleep(1)

if __name__ == '__main__':
    main()
