ROBOT_GAZEBO = "simulation"
ROBOT_UR_MODERN_DRIVER = "ur_modern_driver"
ROBOT_UR_RTDE_DRIVER = "ur_rtde_driver"

JOINT_PUBLISHER_REAL  = 'vel_based_pos_traj_controller'
JOINT_PUBLISHER_BETA  = 'pos_based_pos_traj_controller'
JOINT_PUBLISHER_SIM  = 'arm_controller'
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

JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
