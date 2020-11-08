ROBOT_GAZEBO = "simulation"
ROBOT_GAZEBO_DUAL_RIGHT = "sim_dual_right"
ROBOT_GAZEBO_DUAL_LEFT = "sim_dual_left"
ROBOT_UR_MODERN_DRIVER = "ur_modern_driver"
ROBOT_UR_RTDE_DRIVER = "ur_rtde_driver"

IKFAST = 'ikfast'
TRAC_IK = 'trac_ik'

JOINT_PUBLISHER_REAL = 'vel_based_pos_traj_controller'
JOINT_PUBLISHER_BETA = 'scaled_pos_joint_traj_controller'
JOINT_PUBLISHER_SIM = 'arm_controller'
JOINT_PUBLISHER_SIM_DUAL_RIGHT = 'rightarm/'
JOINT_PUBLISHER_SIM_DUAL_LEFT = 'leftarm/'

JOINT_SUBSCRIBER = '/arm_controller/state'
JOINT_STATE_SUBSCRIBER = 'joint_states'
FT_SUBSCRIBER_REAL = '/wrench'
FT_SUBSCRIBER_SIM = '/ft_sensor/raw'

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

JOINT_ORDER = [
    SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT, WRIST_1_JOINT,
    WRIST_2_JOINT, WRIST_3_JOINT
]

# RESULT_CODE
DONE = 'done'
FORCE_TORQUE_EXCEEDED = 'force_exceeded'
IK_NOT_FOUND = 'ik_not_found'
SPEED_LIMIT_EXCEEDED = 'speed_limit_exceeded'