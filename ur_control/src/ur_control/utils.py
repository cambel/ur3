# ROS utilities used by the CRI group
#! /usr/bin/env python
import os
import sys
import copy
import time
import numpy as np
import rospy
import rospkg
import sys
import inspect
from ur_control import transformations, spalg
from sensor_msgs.msg import JointState
from pyquaternion import Quaternion

def load_urdf_string(package, filename):
    rospack = rospkg.RosPack()
    package_dir = rospack.get_path(package)
    urdf_file = package_dir + '/urdf/' + filename + '.urdf'
    urdf = None
    with open(urdf_file) as f:
        urdf = f.read()
    return urdf

class PDRotation:
    def __init__(self, kp, kd=None):
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.reset()

    def reset(self):
        self.last_time = rospy.get_rostime()
        self.last_error = Quaternion()

    def set_gains(self, kp=None, kd=None):
        if kp is not None:
            self.kp = np.array(kp)
        if kd is not None:
            self.kd = np.array(kd)

    def update(self, quaternion_error, dt=None):
        now = rospy.get_rostime()
        if dt is None:
            dt = now - self.last_time

        k_prime = 2 * quaternion_error.scalar*np.identity(3)-spalg.skew(quaternion_error.vector)
        p_term = np.dot(self.kp, k_prime)

        # delta_error = quaternion_error - self.last_error
        w = transformations.angular_velocity_from_quaternions(quaternion_error, self.last_error, dt)
        d_term = self.kd * w

        output = p_term + d_term
        # Save last values
        self.last_error = quaternion_error
        self.last_time = now
        return output

class PID:
    def __init__(self, Kp, Ki=None, Kd=None, dynamic_pid=False, max_gain_multiplier=200.0):
        # Proportional gain
        self.Kp = np.array(Kp)
        self.Ki = np.zeros_like(Kp)
        self.Kd = np.zeros_like(Kp)
        # Integral gain
        if Ki is not None:
            self.Ki = np.array(Ki)
        # Derivative gain
        if Kd is not None:
            self.Kd = np.array(Kd)
        self.set_windup(np.ones_like(self.Kp))
        # Reset
        self.reset()
        self.dynamic_pid = dynamic_pid
        self.max_gain_multiplier = max_gain_multiplier

    def reset(self):
        self.last_time = rospy.get_rostime()
        self.last_error = np.zeros_like(self.Kp)
        self.integral = np.zeros_like(self.Kp)

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None:
            self.Kp = np.array(Kp)
        if Ki is not None:
            self.Ki = np.array(Ki)
        if Kd is not None:
            self.Kd = np.array(Kd)

    def set_windup(self, windup):
        self.i_min = -np.array(windup)
        self.i_max = np.array(windup)

    def update(self, error, dt=None):
        # CAUTION: naive scaling of the Kp parameter based on the error
        # The main idea, the smaller the error the higher the gain
        if self.dynamic_pid:
            kp = np.abs([self.Kp[i]/error[i] if error[i] != 0.0 else self.Kp[i] for i in range(6)])
            kp = np.clip(kp, self.Kp, self.Kp*self.max_gain_multiplier)
            kd = np.abs([self.Kd[i]*error[i] if error[i] != 0.0 else self.Kd[i] for i in range(6)])
            kd = np.clip(kd, self.Kd/self.max_gain_multiplier, self.Kd)
            ki = self.Ki
        else:
            kp = self.Kp
            kd = self.Kd
            ki = self.Ki

        now = rospy.get_rostime()
        if dt is None:
            dt = now - self.last_time
        delta_error = error - self.last_error
        # Compute terms
        self.integral += error * dt
        p_term = kp * error
        i_term = ki * self.integral
        i_term = np.maximum(self.i_min, np.minimum(i_term, self.i_max))
        
        # First delta error is huge since it was initialized at zero first, avoid considering
        if not np.allclose(self.last_error, np.zeros_like(self.last_error)):
            d_term = kd * delta_error / dt
        else:
            d_term = kd * np.zeros_like(delta_error) / dt

        output = p_term + i_term + d_term
        # Save last values
        self.last_error = np.array(error)
        self.last_time = now
        return output


class TextColors:
    """
    The C{TextColors} class is used as alternative to the C{rospy} logger. It's useful to
    print messages when C{roscore} is not running.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    log_level = rospy.INFO

    def disable(self):
        """
        Resets the coloring.
        """
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

    def blue(self, msg):
        """
        Prints a B{blue} color message
        @type  msg: string
        @param msg: the message to be printed.
        """
        print((self.OKBLUE + msg + self.ENDC))

    def debug(self, msg):
        """
        Prints a B{green} color message
        @type  msg: string
        @param msg: the message to be printed.
        """
        print((self.OKGREEN + msg + self.ENDC))

    def error(self, msg):
        """
        Prints a B{red} color message
        @type  msg: string
        @param msg: the message to be printed.
        """
        print((self.FAIL + msg + self.ENDC))

    def ok(self, msg):
        """
        Prints a B{green} color message
        @type  msg: string
        @param msg: the message to be printed.
        """
        print((self.OKGREEN + msg + self.ENDC))

    def warning(self, msg):
        """
        Prints a B{yellow} color message
        @type  msg: string
        @param msg: the message to be printed.
        """
        print((self.WARNING + msg + self.ENDC))

    def logdebug(self, msg):
        """
        Prints message with the word 'Debug' in green at the begging.
        Alternative to C{rospy.logdebug}.
        @type  msg: string
        @param msg: the message to be printed.
        """
        if self.log_level <= rospy.DEBUG:
            print((self.OKGREEN + 'Debug ' + self.ENDC + str(msg)))

    def loginfo(self, msg):
        """
        Prints message with the word 'INFO' begging.
        Alternative to C{rospy.loginfo}.
        @type  msg: string
        @param msg: the message to be printed.
        """
        if self.log_level <= rospy.INFO:
            print(('INFO ' + str(msg)))

    def logwarn(self, msg):
        """
        Prints message with the word 'Warning' in yellow at the begging.
        Alternative to C{rospy.logwarn}.
        @type  msg: string
        @param msg: the message to be printed.
        """
        if self.log_level <= rospy.WARN:
            print((self.WARNING + 'Warning ' + self.ENDC + str(msg)))

    def logerr(self, msg):
        """
        Prints message with the word 'Error' in red at the begging.
        Alternative to C{rospy.logerr}.
        @type  msg: string
        @param msg: the message to be printed.
        """
        if self.log_level <= rospy.ERROR:
            print((self.FAIL + 'Error ' + self.ENDC + str(msg)))

    def logfatal(self, msg):
        """
        Prints message with the word 'Fatal' in red at the begging.
        Alternative to C{rospy.logfatal}.
        @type  msg: string
        @param msg: the message to be printed.
        """
        if self.log_level <= rospy.FATAL:
            print((self.FAIL + 'Fatal ' + self.ENDC + str(msg)))

    def set_log_level(self, level):
        """
        Sets the log level. Possible values are:
          - DEBUG:  1
          - INFO:   2
          - WARN:   4
          - ERROR:  8
          - FATAL:  16
        @type  level: int
        @param level: the new log level
        """
        self.log_level = level


## Helper Functions ##
def assert_shape(variable, name, shape):
    """
    Asserts the shape of an np.array
    @type  variable: Object
    @param variable: variable to be asserted
    @type  name: string
    @param name: variable name
    @type  shape: tuple
    @param ttype: expected shape of the np.array
    """
    assert variable.shape == shape, '%s must have a shape %r: %r' % (name, shape, variable.shape)


def assert_type(variable, name, ttype):
    """
    Asserts the type of a variable with a given name
    @type  variable: Object
    @param variable: variable to be asserted
    @type  name: string
    @param name: variable name
    @type  ttype: Type
    @param ttype: expected variable type
    """
    assert type(variable) is ttype,  '%s must be of type %r: %r' % (name, ttype, type(variable))


def db_error_msg(name, logger=TextColors()):
    """
    Prints out an error message appending the given database name.
    @type  name: string
    @param name: database name
    @type  logger: Object
    @param logger: Logger instance. When used in ROS, the recommended C{logger=rospy}.
    """
    msg = 'Database %s not found. Please generate it. [rosrun denso_openrave generate_databases.py]' % name
    logger.logerr(msg)


def clean_cos(value):
    """
    Limits the a value between the range C{[-1, 1]}
    @type value: float
    @param value: The input value
    @rtype: float
    @return: The limited value in the range C{[-1, 1]}
    """
    return min(1, max(value, -1))


def has_keys(data, keys):
    """
    Checks whether a dictionary has all the given keys.
    @type   data: dict
    @param  data: Parameter name
    @type   keys: list
    @param  keys: list containing the expected keys to be found in the dict.
    @rtype: bool
    @return: True if all the keys are found in the dict, false otherwise.
    """
    if not isinstance(data, dict):
        return False
    has_all = True
    for key in keys:
        if key not in data:
            has_all = False
            break
    return has_all


def raise_not_implemented():
    """
    Raises a NotImplementedError exception
    """
    raise NotImplementedError()


def read_key(echo=False):
    """
    Reads a key from the keyboard
    @type   echo: bool, optional
    @param  echo: if set, will show the input key in the console.
    @rtype: str
    @return: The limited value in the range C{[-1, 1]}
    """
    if not echo:
        os.system("stty -echo")
    key = sys.stdin.read(1)
    if not echo:
        os.system("stty echo")
    return key.lower()


def read_parameter(name, default):
    """
    Get a parameter from the ROS parameter server. If it's not found, a
    warn is printed.
    @type  name: string
    @param name: Parameter name
    @type  default: Object
    @param default: Default value for the parameter. The type should be
    the same as the one expected for the parameter.
    @rtype: any
    @return: The resulting parameter
    """
    if rospy.is_shutdown():
        logger = TextColors()
        logger.logwarn('roscore not found, parameter [%s] using default: %s' % (name, default))
    else:
        if not rospy.has_param(name):
            rospy.logwarn('Parameter [%s] not found, using default: %s' % (name, default))
        return rospy.get_param(name, default)
    return default


def read_parameter_err(name):
    """
    Get a parameter from the ROS parameter server. If it's not found, a
    error is printed.
    @type name: string
    @param name: Parameter name
    @rtype: has_param, param
    @return: (has_param) True if succeeded, false otherwise. The
    parameter is None if C{has_param=False}.
    """
    if rospy.is_shutdown():
        logger = TextColors()
        logger.logerr('roscore not found')
        has_param = False
    else:
        has_param = True
        if not rospy.has_param(name):
            rospy.logerr("Parameter [%s] not found" % (name))
            has_param = False
    return has_param, rospy.get_param(name, None)


def read_parameter_fatal(name):
    """
    Get a parameter from the ROS parameter server. If it's not found, an
    exception will be raised.
    @type name: string
    @param name: Parameter name
    @rtype: any
    @return: The resulting parameter
    """
    if rospy.is_shutdown():
        logger = TextColors()
        logger.logfatal('roscore not found')
        raise Exception('Required parameter {0} not found'.format(name))
    else:
        if not rospy.has_param(name):
            rospy.logfatal("Parameter [%s] not found" % (name))
            raise Exception('Required parameter {0} not found'.format(name))
    return rospy.get_param(name, None)


def solve_namespace(namespace=''):
    """
    Appends neccessary slashes required for a proper ROS namespace.
    @type namespace: string
    @param namespace: namespace to be fixed.
    @rtype: string
    @return: Proper ROS namespace.
    """
    if len(namespace) == 0:
        namespace = rospy.get_namespace()
    elif len(namespace) == 1:
        if namespace != '/':
            namespace = '/' + namespace + '/'
    else:
        if namespace[0] != '/':
            namespace = '/' + namespace
        if namespace[-1] != '/':
            namespace += '/'
    return namespace


def sorted_joint_state_msg(msg, joint_names):
    """
    Returns a sorted C{sensor_msgs/JointState} for the given joint names
    @type  msg: sensor_msgs/JointState
    @param msg: The input message
    @type  joint_names: list
    @param joint_names: The sorted joint names
    @rtype: sensor_msgs/JointState
    @return: The C{JointState} message with the fields in the order given by joint names
    """
    valid_names = set(joint_names).intersection(set(msg.name))
    valid_position = len(msg.name) == len(msg.position)
    valid_velocity = len(msg.name) == len(msg.velocity)
    valid_effort = len(msg.name) == len(msg.effort)
    num_joints = len(valid_names)
    retmsg = JointState()
    retmsg.header = copy.deepcopy(msg.header)
    for name in joint_names:
        if name not in valid_names:
            continue
        idx = msg.name.index(name)
        retmsg.name.append(name)
        if valid_position:
            retmsg.position.append(msg.position[idx])
        if valid_velocity:
            retmsg.velocity.append(msg.velocity[idx])
        if valid_effort:
            retmsg.effort.append(msg.effort[idx])
    return retmsg


def unique(data):
    """
    Finds the unique elements of an array. B{row-wise} and
    returns the sorted unique elements of an array.
    @type  data: np.array
    @param data: Input array.
    @rtype: np.array
    @return: The sorted unique array.
    """
    order = np.lexsort(data.T)
    data = data[order]
    diff = np.diff(data, axis=0)
    ui = np.ones(len(data), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return data[ui]


def wait_for(predicate, timeout=5.0):
    start_time = time.time()
    while not predicate():
        now = time.time()
        if (now - start_time) > timeout:
            return False
        time.sleep(0.001)
    return True
