JOINT_PUBLISHER_REAL  = '/vel_based_pos_traj_controller/command'
JOINT_PUBLISHER_SIM  = '/arm_controller/command'
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
