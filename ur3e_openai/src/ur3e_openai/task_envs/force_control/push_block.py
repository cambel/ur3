
import rospy
import numpy as np
from ur3e_openai.robot_envs.utils import get_board_color

from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import transformations, spalg, conversions
from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_gazebo.basic_models import get_box_model
from ur_gazebo.model import Model


class UR3ePushBlockEnv(UR3eForceControlEnv):
    """ Push block with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        string_model = get_box_model("cube", self.block_size, color=[0, 0, 1, 0],
                                     mass=self.block_mass, mu=self.block_friction, kp=self.block_stiffness)
        self.box_model = Model("block", self.block_initial_pose, file_type="string",
                               string_model=string_model, model_id="target_block")

        self.world_to_robot_base = transformations.pose_to_transform(
            [0, 0, -0.7, 0, 0, np.pi/2])

        self.stage = 0

    def __load_env_params(self):
        prefix = "ur3e_gym"
        # Gazebo spawner parameters
        self.randomize_block_properties = rospy.get_param(
            prefix + "/randomize_block_properties", False)
        self.block_initial_pose = rospy.get_param(
            prefix + "/block_initial_pose")
        self.block_size = rospy.get_param(prefix + "/block_size", 0.05)
        self.block_mass = rospy.get_param(prefix + "/block_mass", 1.0)
        self.block_friction = rospy.get_param(prefix + "/block_friction", 1)
        self.block_stiffness = rospy.get_param(
            prefix + "/block_stiffness", 1e5)
        # How often to generate a new model, number of episodes
        self.refresh_rate = rospy.get_param(prefix + "/refresh_rate", False)

    def update_scene(self):
        self.stage = 0
        block_pose = self.randomize_block_position()
        if self.randomize_block_properties:
            randomize_value = self.rng.uniform(size=3)
            mass = np.interp(randomize_value[0], [-1., 1.], [0.5, 5.])
            friction = np.interp(randomize_value[0], [-1., 1.], [0, 5.])
            stiffness = np.interp(randomize_value[0], [-1., 1.], [1e4, 1.e8])
            color = list(get_board_color(stiffness=stiffness, stiff_lower=1e4, stiff_upper=1e8))+[0]
            string_model = get_box_model("cube", self.block_size, color=color,
                                        mass=mass, mu=friction, kp=stiffness)
            self.box_model = Model("block", block_pose, file_type="string",
                                   string_model=string_model, model_id="target_block")
            self.spawner.load_models([self.box_model])
        else:
            self.box_model.set_pose(block_pose)
            self.spawner.update_model_state(self.box_model)

        t_btn_pose = transformations.pose_to_transform(block_pose)
        t_w2tcp = transformations.concatenate_matrices(
            self.world_to_robot_base, t_btn_pose)
        self.current_target_pose = transformations.pose_quaternion_from_matrix(
            t_w2tcp)
        self.current_target_pose[3:] = self.target_pose[3:]

        self.target_model.set_pose(self.current_target_pose)
        self.spawner.update_model_state(self.target_model)

    def randomize_block_position(self):
        rand = self.rng.random(size=2)
        rand = np.array([np.interp(rand[i], [0, 1.], self.workspace[i])
                        for i in range(2)])
        pose = np.copy(self.block_initial_pose)
        pose[:2] += rand  # only x and y
        return pose

    def _is_done(self, observations):
        pose_error = observations[:6]
        position_reached = np.all(
            np.abs(pose_error[:3]) < self.position_threshold)
        orientation_reached = np.all(
            np.abs(pose_error[3:]) < self.orientation_threshold)
        if position_reached and orientation_reached and self.stage == 0:
            rospy.loginfo("First stage reach")
            self.current_target_pose[2] -= 0.10
            self.stage = 1
            return False
        return (position_reached and orientation_reached and self.stage == 1) \
            or self.action_result == FORCE_TORQUE_EXCEEDED  # Stop on collision
