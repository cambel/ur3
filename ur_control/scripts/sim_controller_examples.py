#!/usr/bin/env python
from ur_control import utils, spalg, transformations, traj_utils
from ur_control.hybrid_controller import ForcePositionController
from ur_control.compliant_controller import CompliantController
import argparse
import rospy
import timeit
import numpy as np
from pyquaternion import Quaternion
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def move_joints(wait=True):
    # desired joint configuration 'q'
    # q = [2.37191, -1.88688, -1.82035,  0.4766,  2.31206,  3.18758]
    q = [1.0884, -1.4911, 1.9667, -2.0092, -1.5708, -0.4378]
    # go to desired joint configuration
    # in t time (seconds)
    # wait is for waiting to finish the motion before executing
    # anything else or ignore and continue with whatever is next
    arm.set_joint_positions(position=q, wait=wait, t=0.5)


def follow_trajectory():
    traj = [
        [2.4463, -1.8762, -1.6757, 0.3268, 2.2378, 3.1960],
        [2.5501, -1.9786, -1.5293, 0.2887, 2.1344, 3.2062],
        [2.5501, -1.9262, -1.3617, 0.0687, 2.1344, 3.2062],
        [2.4463, -1.8162, -1.5093, 0.1004, 2.2378, 3.1960],
        [2.3168, -1.7349, -1.6096, 0.1090, 2.3669, 3.1805],
        [2.3168, -1.7997, -1.7772, 0.3415, 2.3669, 3.1805],
        [2.3168, -1.9113, -1.8998, 0.5756, 2.3669, 3.1805],
        [2.4463, -1.9799, -1.7954, 0.5502, 2.2378, 3.1960],
        [2.5501, -2.0719, -1.6474, 0.5000, 2.1344, 3.2062],
    ]
    for t in traj:
        arm.set_joint_positions(position=t, wait=True, t=1.0)


def move_endeffector(wait=True):
    # get current position of the end effector
    cpose = arm.end_effector()
    # define the desired translation/rotation
    deltax = np.array([0., 0., 0.04, 0., 0., 0.])
    # add translation/rotation to current position
    cpose = transformations.pose_euler_to_quaternion(cpose, deltax, ee_rotation=True)
    # execute desired new pose
    # may fail if IK solution is not found
    arm.set_target_pose(pose=cpose, wait=True, t=1.0)


def move_gripper():
    print("closing")
    arm.gripper.close()
    rospy.sleep(1.0)
    print("opening")
    arm.gripper.open()
    rospy.sleep(1.0)
    print("moving")
    arm.gripper.command(0.5, percentage=True)  # in percentage (80%)
    # 0.0 is full close, 1.0 is full open
    rospy.sleep(1.0)
    print("moving")
    arm.gripper.command(0.01)  # in meters
    # 0.05 is full open, 0.0 is full close
    # max gap for the Robotiq Hand-e is 0.05 meters

    print("current gripper position", round(arm.gripper.get_position(), 4), "meters")


def grasp_naive():
    # probably won't work
    arm.gripper.open()
    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)

    q2 = [1.82225, -1.55525,  1.86741, -2.03039, -1.60938,  0.24935]
    arm.set_joint_positions(q2, wait=True, t=1.0)

    arm.gripper.command(0.036)
    rospy.sleep(0.5)

    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)


def grasp_plugin():
    arm.gripper.open()
    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)

    q2 = [1.82225, -1.55525,  1.86741, -2.03039, -1.60938,  0.24935]
    arm.set_joint_positions(q2, wait=True, t=1.0)

    arm.gripper.command(0.039)
    # attach the object "link" to the robot "model_name"::"link_name"
    arm.gripper.grab(link_name="cube3::link")

    q1 = [1.82224, -1.59475,  1.68247, -1.80611, -1.60922,  0.24936]
    arm.set_joint_positions(q1, wait=True, t=1.0)
    rospy.sleep(2.0)  # release after 2 secs

    # dettach the object "link" to the robot "model_name"::"link_name"
    arm.gripper.open()
    arm.gripper.release(link_name="cube3::link")


def circular_trajectory():
    initial_q = [5.1818, 0.3954, -1.5714, -0.3902, 1.5448, -0.5056]
    arm.set_joint_positions(initial_q, wait=True, t=2)

    target_position = [0.20, -0.30, 0.80]
    target_orienation = [0.01504, 0.01646, -0.01734, 0.9996]
    target_orienation = [0.18603, 0.01902, -0.01464, 0.98225]
    target_pose = target_position + target_orienation

    deltax = np.array([0., 0.05, 0.1, 0., 0., 0.])
    initial_pose = transformations.pose_euler_to_quaternion(target_pose, deltax, ee_rotation=True)

    initial_pose = initial_pose[:3]
    final_pose = target_position[:3]

    target_q = transformations.vector_to_pyquaternion(target_orienation)

    p1 = target_q.rotate(initial_pose - final_pose)
    p2 = np.zeros(3)
    steps = 20
    duration = 10

    traj = traj_utils.circunference(p1, p2, steps)
    traj = np.apply_along_axis(target_q.rotate, 1, traj)
    trajectory = traj + final_pose

    for position in trajectory:
        cmd = np.concatenate([position, target_orienation])
        arm.set_target_pose(cmd, wait=True, t=(duration/steps))


def face_towards_target():
    """
        Move robot's end-effector towards a target point. 
    """
    cpose = arm.end_effector()  # current pose
    target_position = [0.35951, -0.54521, 0.34393]
    # compute pose with new rotation
    cmd = spalg.face_towards(target_position, cpose)
    arm.set_target_pose(cmd, wait=True, t=1)

def force_control():
    arm.set_wrench_offset(True)
    alpha = [1.,1.,0.0,1.,1.,1.]
    robot_control_dt = 0.002

    timeout = 20.0

    Kp = np.array([0.01]*6)
    Kp_pos = Kp
    Kd_pos = Kp * 0.1
    position_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

    # Force PID gains
    Kp = np.array([0.05]*6)
    Kp_force = Kp
    Kd_force = Kp * 0.1
    Ki_force = Kp * 0.01
    force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
    pf_model = ForcePositionController(position_pd=position_pd, force_pd=force_pd, alpha=np.diag(alpha), dt=robot_control_dt)

    max_force_torque = np.array([50.,50.,50.,5.,5.,5.])

    target_position = arm.end_effector()
    target_force = np.array([0.,0.,0.,0.,0.,0.])

    pf_model.set_goals(target_position, target_force)

    res = arm.set_hybrid_control(pf_model, max_force_torque=max_force_torque, timeout=timeout)
    print(res)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-t', '--move_traj', action='store_true',
                        help='move following a trajectory of joint configurations')
    parser.add_argument('-e', '--move_ee', action='store_true',
                        help='move to a desired end-effector position')
    parser.add_argument('-g', '--gripper', action='store_true',
                        help='Move gripper')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force control demo')
    parser.add_argument('--grasp_naive', action='store_true',
                        help='Test simple grasping (cube_tasks world)')
    parser.add_argument('--grasp_plugin', action='store_true',
                        help='Test grasping plugin (cube_tasks world)')
    parser.add_argument('--circle', action='store_true',
                        help='Circular rotation around a target pose')
    parser.add_argument('--face', action='store_true',
                        help='Face towards a target vector')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm', default=None)
    args = parser.parse_args()

    rospy.init_node('ur3e_script_control')

    tcp_z = 0.0  # where to consider the tool center point wrt end-effector
    if args.face:
        tcp_z = 0.0

    ns = ''
    joints_prefix = None
    robot_urdf = "ur3e_robot"
    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
    
    use_gripper = args.gripper  

    extra_ee = [0, 0, tcp_z, 0, 0, 0, 1]

    global arm
    arm = CompliantController(ft_sensor=True, ee_transform=extra_ee, 
              gripper=use_gripper, namespace=ns, 
              joint_names_prefix=joints_prefix, 
              robot_urdf=robot_urdf)
    print("Extra ee", extra_ee)

    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    if args.move:
        move_joints()
    if args.move_traj:
        follow_trajectory()
    if args.move_ee:
        move_endeffector()
    if args.gripper:
        move_gripper()
    if args.grasp_naive:
        grasp_naive()
    if args.grasp_plugin:
        grasp_plugin()
    if args.circle:
        circular_trajectory()
    if args.face:
        face_towards_target()
    if args.force:
        force_control()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
