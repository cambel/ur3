from ur3e_openai import robot_gazebo_env
import rospy
import numpy as np
from pyquaternion import Quaternion

from ur_control.arm import Arm
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
import ur_control.utils as utils

from gps.utility.general_utils import get_ee_points  # For getting points and velocities.


class UR3eEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """

        rospy.logdebug("Start UR3eBasicEnv Init")
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv

        super(UR3eEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=reset_controls_bool,
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD")

        self.get_params()

        rospy.logdebug("UR3eEnv unpause...")
        self.gazebo.unpauseSim()

        self.ur3e_arm = Arm(ft_sensor=self.ft_sensor, driver=self.driver)

        self._previous_joints = None

        self.gazebo.pauseSim()
        rospy.logdebug("Finished UR3eEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def get_params(self):
        self.ft_sensor = rospy.get_param("/ur3e/ft_sensor")
        driver_param = rospy.get_param("/ur3e/driver")
        self.driver = ROBOT_GAZEBO
        if driver_param == "robot":
            self.driver = ROBOT_UR_MODERN_DRIVER
        elif driver_param == "beta":
            self.driver = ROBOT_UR_RTDE_DRIVER
        self.slowness = rospy.get_param("/ur3e/slowness")
        self.dt = rospy.get_param("ur3e/dt")
        self.init_pos = np.array(rospy.get_param("/ur3e/init_pos"))
        self.target_pos = np.array(rospy.get_param("ur3e/target_pos"))

        self.scale_pos = rospy.get_param("ur3e/scale_pos")
        self.end_effector_points = rospy.get_param("ur3e/end_effector_points")
        self.distance_threshold = rospy.get_param("ur3e/distance_threshold")

        self.n_actions = rospy.get_param("/ur3e/n_actions")
        self.reward_type = rospy.get_param("ur3e/reward_type")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        return True

    def get_points_and_vels(self, joint_angles):
        """
        Helper function that gets the cartesian positions
        and velocities from ROS."""

        if self._previous_joints is None:
            self._previous_joints = self.init_pos

        # Current position
        ee_pos_now = self._get_ee_pose(joint_angles)
        # print ee_pos_now
        # Last position
        ee_pos_last = self._get_ee_pose(self._previous_joints)

        # Use the past position to get the present velocity.
        velocity = (ee_pos_now - ee_pos_last) / self.dt

        # Shift the present poistion by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        position = (np.asarray(ee_pos_now) - np.asarray(
            self.target_pos))[0] * self.scale_pos
        position = np.squeeze(np.asarray(position))

        self._previous_joints = joint_angles  # update

        return position, velocity

    def _get_ee_pose(self, joint_angles):
        EE_POINTS = np.array(self.end_effector_points)

        pose = self.ur3e_arm.kinematics.forward_position_kinematics(
            joint_angles)  #[x, y, z, ax, ay, az, w]
        translation = np.array(pose[:3]).reshape(1, 3)
        rot = Quaternion(np.roll(pose[3:], 1)).rotation_matrix

        return np.ndarray.flatten(
            get_ee_points(EE_POINTS,
                          np.array(translation).reshape((1, 3)), rot).T)

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------