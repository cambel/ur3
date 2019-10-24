import rospy
import numpy as np

from gym import spaces

from ur3e_openai.robot_envs import ur3e_ee_env
from ur3e_openai.task_envs.task_commons import load_ros_params


class UR3ePegInHoleEnv(ur3e_ee_env.UR3eEEEnv):
    def __init__(self):

        ur3e_ee_env.UR3eEEEnv.__init__(self)
        
        # Load Params from the desired Yaml file
        load_ros_params(rospackage_name="ur3e_openai",
                               rel_path_from_package_to_file="src/ur3e_openai/task_envs/ur3e/config",
                               yaml_file_name="peg_in_hole.yaml")
        self.get_params()

        self._init_hybrid_controller(self.control, self.xF_tradeoff)

        obs = self._get_obs()

        self.reward_threshold = -700.0

        self.action_space = spaces.Box(-1.,
                                       1.,
                                       shape=(self.n_actions, ),
                                       dtype='float32')

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs.shape,
                                            dtype='float32')
        # self.observation_space = spaces.Dict(dict(
        #     desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        # ))

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

    def get_params(self):
        self.control = rospy.get_param("ur3e/control")
        self.xF_tradeoff = rospy.get_param("ur3e/xF_tradeoff")

        self.init_pos = np.array(rospy.get_param("/ur3e/init_pos"))
        self.target_pos = np.array(rospy.get_param("ur3e/target_pos"))

        self.tgt_pose_indices = rospy.get_param("ur3e/tgt_pose_indices")

        self.scale_pos = rospy.get_param("ur3e/scale_pos")
        self.end_effector_points = rospy.get_param("ur3e/end_effector_points")
        self.distance_threshold = rospy.get_param("ur3e/distance_threshold")

        self.n_actions = rospy.get_param("/ur3e/n_actions")
        self.reward_type = rospy.get_param("ur3e/reward_type")

        self.pose_indices = rospy.get_param("ur3e/pose_indices")

        self.cost_l1 = rospy.get_param("ur3e/cost/l1")
        self.cost_l2 = rospy.get_param("ur3e/cost/l2")
        self.cost_alpha = rospy.get_param("ur3e/cost/alpha")

        self.trials = 2

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.controller.reset()
        self.ur3e_arm.set_joint_positions(position=self.init_pos,
                                          wait=True,
                                          t=2)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # TODO

    def _set_action(self, action):

        if isinstance(action, int):
            acts = np.zeros(int(self.n_actions / 2))
            if action < self.n_actions / 2:
                acts[action] += 0.1
            else:
                acts[int(action - self.n_actions / 2)] -= 0.01
        elif isinstance(action, np.ndarray):
            assert action.shape == (self.n_actions, )
            # ensure that we don't change the action outside of this scope
            acts = action.copy() / 5.

        # print ("CB action", acts)
        # Take action
        if np.any(np.isnan(acts)):
            rospy.logerr("Invalid NAN action(s)" + str(acts))
            return

        self._set_target_pose(acts)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        joint_angles = self.ur3e_arm.joint_angles()
        # joint_velocities = self.ur3e_arm.joint_velocities()
        self.ee_points, ee_velocities = self.get_points_and_vels(joint_angles)

        obs = np.concatenate([
            # joint_angles.ravel(),
            # joint_velocities.ravel(),
            self.ee_points.ravel(),
            ee_velocities.ravel()
        ])

        return obs.copy()
        # return {
        #     'observation': obs.copy(),
        #     'achieved_goal': ee_points.copy(),
        #     'desired_goal': self.target_pos.copy(),
        # }

    def _is_done(self, observations):
        target_pos = np.zeros_like(self.ee_points)
        d = self.goal_distance(self.ee_points, target_pos)
        self._log_message = "Final distance: " + str(d)
        return (d < self.distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        target_pos = np.zeros_like(self.ee_points)
        d = self.goal_distance(self.ee_points, target_pos)

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'l1l2':
            return -(0.5 * (d ** 2) * self.cost_l2 + \
                np.sqrt(self.cost_alpha + (d ** 2)) * self.cost_l1)
        else:
            return -d

    def _env_setup(self, initial_qpos):
        self.gazebo.unpauseSim()
        self.ur3e_arm.set_joint_positions(position=self.init_pos,
                                          t=2,
                                          wait=True)

    def get_points_and_vels(self, joint_angles):
        """
        Helper function that gets the cartesian positions
        and velocities from ROS."""

        if self._previous_joints is None:
            self._previous_joints = self.init_pos

        # Current position
        ee_pos_now = self.ur3e_arm.end_effector(q=joint_angles,
                                                rot_type='euler')

        # Last position
        ee_pos_last = self.ur3e_arm.end_effector(q=self._previous_joints,
                                                 rot_type='euler')
        self._previous_joints = joint_angles  # update

        # Use the past position to get the present velocity.
        velocity = (ee_pos_now - ee_pos_last) / self.dt

        # Shift the present poistion by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        position = (np.asarray(ee_pos_now) - np.asarray(self.target_pos))

        position = np.squeeze(np.asarray(position))
        # wrap rotation values between [pi,-pi]
        position[3:] = (position[3:] + np.pi) % (2 * np.pi) - np.pi
        # scale position error, for more precise motion
        # (numerical error with small numbers?)
        position *= [1000, 1000, 1000, 100., 100., 10.]

        # Extract only positions of interest
        if self.tgt_pose_indices:
            position = np.array([position[i] for i in self.tgt_pose_indices])
            velocity = np.array([velocity[i] for i in self.tgt_pose_indices])

        return position, velocity

    def _set_target_pose(self, action):
        self.controller.position_pd.reset()
        self.controller.force_pd.reset()
        target = self.ur3e_arm.end_effector(rot_type='euler')

        if self.pose_indices:
            target[self.pose_indices] += action[:]
        else:
            target[:] += action[:]

        # print "pose:", action
        # target[:2] = [0,0.5]
        self.controller.set_goals(position=target)

        Fr = np.zeros(6)
        self.controller.set_goals(force=Fr)

        self.controller.start(
            dt=0.002,  # 500hz
            timeout=self.dt,
            controller=self.ur3e_arm)
