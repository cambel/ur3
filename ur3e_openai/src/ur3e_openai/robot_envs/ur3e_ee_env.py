from ur3e_openai import robot_gazebo_env
import rospy
import numpy as np

from ur_control.arm import Arm
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
import ur_control.utils as utils

from gps.agent.ur.force_controller import ForcePositionController, PositionController


class UR3eEEEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """
    def __init__(self):
        """Initializes a new Robot environment.
        """

        rospy.logdebug("Start UR3eEEEnv Init")
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['arm_controller']

        # It doesnt use namespace
        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv

        super(UR3eEEEnv, self).__init__(controllers_list=self.controllers_list,
                                        robot_name_space=self.robot_name_space,
                                        reset_controls=reset_controls_bool,
                                        start_init_physics_parameters=False,
                                        reset_world_or_sim="WORLD")

        self.get_robot_params()

        rospy.logdebug("UR3eEnv unpause...")
        self.gazebo.unpauseSim()

        self.ur3e_arm = Arm(ft_sensor=self.ft_sensor, driver=self.driver)

        self._previous_joints = None

        self.gazebo.pauseSim()
        rospy.logdebug("Finished UR3eEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def get_robot_params(self):
        driver_param = rospy.get_param("/ur3e/driver")
        self.driver = ROBOT_GAZEBO
        if driver_param == "robot":
            self.driver = ROBOT_UR_MODERN_DRIVER
        elif driver_param == "beta":
            self.driver = ROBOT_UR_RTDE_DRIVER
        self.ft_sensor = rospy.get_param("/ur3e/ft_sensor")
        self.dt = rospy.get_param("ur3e/dt")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        return True

    def _init_hybrid_controller(self, control, xF_tradeoff):
        # self.ur3e_arm.set_wrench_offset(override=True)

        # position PD
        # proportional gain of position controller
        Kp = 10  #250
        Kp_pos = np.array([Kp * 1e-6] * 3 + [Kp * 1e-6, Kp * 1e-6, Kp * 1e-6])
        # derivative gain of position controller
        Kd_pos = np.array(
            [np.sqrt(Kp) * 1e-6] * 3 +
            [np.sqrt(Kp) *
             1e-6, np.sqrt(Kp) *
             1e-6, np.sqrt(Kp) * 1e-6])
        x_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

        # Force PD
        # proportional gain of force controller
        Kp = 40  #40
        Kp_force = np.array([Kp * 1e-8] * 5 + [-1e-8 * Kp])
        # derivative gain of force controller
        Kd_force = np.array([Kp * 5.0e-10] * 5 + [-1e-8 * Kp])
        F_pd = utils.PID(Kp=Kp_force, Kd=Kd_force)

        if control == "pose":
            self.controller = PositionController(Kp=Kp_pos, Kd=Kd_pos)

        if control == "force_control":
            self.controller = ForcePositionController(
                x_PID=x_pd, F_PID=F_pd, xF_tradeoff=np.diag(xF_tradeoff))

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