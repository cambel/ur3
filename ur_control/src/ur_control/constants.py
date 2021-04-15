IKFAST = 'ikfast'
TRAC_IK = 'trac_ik'

JOINT_PUBLISHER_ROBOT = 'scaled_pos_joint_traj_controller'

JOINT_SUBSCRIBER = '/arm_controller/state'
JOINT_STATE_SUBSCRIBER = 'joint_states'
FT_SUBSCRIBER = 'wrench'

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

BASE_LINK = 'base_link'
EE_LINK = 'wrist_3_link'

JOINT_ORDER = [
    SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT, WRIST_1_JOINT,
    WRIST_2_JOINT, WRIST_3_JOINT
]

def get_arm_joint_names(prefix):
    return [prefix + joint for joint in JOINT_ORDER]

# RESULT_CODE
DONE = 'done'
FORCE_TORQUE_EXCEEDED = 'force_exceeded'
IK_NOT_FOUND = 'ik_not_found'
SPEED_LIMIT_EXCEEDED = 'speed_limit_exceeded'
