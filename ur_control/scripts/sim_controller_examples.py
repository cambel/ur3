#!/usr/bin/env python
import sys
import signal
from ur_control import utils, spalg, transformations, traj_utils
from ur_control.constants import FORCE_TORQUE_EXCEEDED
from ur_control.hybrid_controller import ForcePositionController
from ur_control.compliant_controller import CompliantController
import argparse
import rospy
import rospkg
import yaml
import timeit
import numpy as np
from pyquaternion import Quaternion
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def move_joints(wait=True):
    # desired joint configuration 'q'
    # q = [2.37191, -1.88688, -1.82035,  0.4766,  2.31206,  3.18758]
    q = [1.5701, -1.1854, 1.3136, -1.6975, -1.5708, -0.0016]
    q = [1.5794, -1.4553, 2.1418, -2.8737, -1.6081, 0.0063]
    q = [1.7321, -1.4295, 2.0241, -2.6473, -1.6894, -1.4177]
    q = [1.6626, -1.2571, 1.9806, -2.0439, -2.7765, -1.3049]  # b_bot bearing
    # q = [1.6241, -1.2576, 2.0085, -2.1514, -2.7841, -1.408] # b_bot grasp bearing
    q = [1.6288, -1.3301, 1.8391, -2.0612, -1.5872, -1.5548]  # b_bot
    q = [1.5837, -1.2558, 1.826, -2.194, -2.6195, -1.5081]  # push
    # go to desired joint configuration
    # in t time (seconds)
    # wait is for waiting to finish the motion before executing
    # anything else or ignore and continue with whatever is next
    arm.set_joint_positions(position=q, wait=wait, t=2.0)


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
    deltax = np.array([0., 0., 0.0, 0., 1., 0.])
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


def move_to_pose():
    cpose = arm.end_effector()
    cpose[3:] = [0, 0, 0, 1]
    arm.set_target_pose(pose=cpose, wait=True, t=1.0)

    # def _conical_helix_trajectory(self, steps, revolutions):
    #     # initial_pose = self.ur3e_arm.end_effector()[:3]
    #     initial_pose = self.rand_init_cpose[:3]
    #     final_pose = self.target_pos[:3]

    #     target_q = transformations.vector_to_pyquaternion(self.target_pos[3:])

    #     p1 = target_q.rotate(initial_pose - final_pose)
    #     p2 = np.zeros(3)

    #     traj = get_conical_helix_trajectory(p1, p2, steps, revolutions)
    #     traj = np.apply_along_axis(target_q.rotate, 1, traj)
    #     self.base_trajectory = traj + final_pose


def spiral_trajectory():
    initial_q = [1.6626, -1.2571, 1.9806, -2.0439, -2.7765, -1.3049]  # b_bot bearing
    initial_q = [1.7095, -1.5062, 2.0365, -1.8598, -2.6038, -1.3207]  # b_bot shaft
    initial_q = [1.6463, -1.2494, 1.7844, -2.0497, -2.6194, -1.3827]  # push # b_bot bearing with housing

    arm.set_joint_positions(initial_q, wait=True, t=2)

    plane = "YZ"
    radius = 0.003
    radius_direction = "+Z"
    revolutions = 2

    steps = 100
    duration = 5.0

    arm.set_wrench_offset(True)

    for _ in range(1):
        initial_pose = arm.end_effector()
        trajectory = traj_utils.compute_trajectory(initial_pose, plane, radius, radius_direction, steps, revolutions, trajectory_type="spiral", from_center=True,
                                                   wiggle_direction="Y", wiggle_angle=np.deg2rad(1.0), wiggle_revolutions=10.0)
        execute_trajectory(trajectory, duration=duration, use_force_control=True)


def circular_trajectory():
    initial_q = [1.6626, -1.2571, 1.9806, -2.0439, -2.7765, -1.3049]  # b_bot bearing
    arm.set_joint_positions(initial_q, wait=True, t=2)

    plane = "YZ"
    radius = 0.003
    radius_direction = "+Y"

    steps = 200
    revolutions = 2
    duration = 5.0

    arm.set_wrench_offset(True)

    for _ in range(1):  # Execute the trajectory twice starting from the end of the previous trajectory
        initial_pose = arm.end_effector()
        trajectory = traj_utils.compute_trajectory(initial_pose, plane, radius, radius_direction, steps, revolutions, trajectory_type="circular", from_center=False,
                                                   wiggle_direction="X", wiggle_angle=np.deg2rad(0.0), wiggle_revolutions=10.0)
        execute_trajectory(trajectory, duration=duration, use_force_control=True)


def test_multiple_planes():
    planes = ["XY", "XZ", "YZ"]
    radius_directions = [["+X", "-X", "+Y", "-Y"], ["+X", "-X", "+Z", "-Z"], ["+Y", "-Y", "+Z", "-Z"]]
    for a, b in zip(planes, radius_directions):
        for r in b:
            print("PLANE", a, r)
            initial_q = [1.5909, -1.3506, 1.9397, -2.0634, -2.5136, -1.4549]  # b_bot bearing
            arm.set_joint_positions(initial_q, wait=True, t=1)

            plane = a
            radius = 0.05
            radius_direction = r

            steps = 100
            duration = 2.0

            initial_pose = arm.end_effector()
            trajectory = traj_utils.compute_trajectory(initial_pose, plane, radius, radius_direction, steps, revolutions=1, trajectory_type="spiral", from_center=False)
            execute_trajectory(trajectory, duration=duration, use_force_control=True)


def wiggle():
    initial_pose = arm.end_effector()
    steps = 100.
    traj = traj_utils.compute_rotation_wiggle(initial_pose[3:], "Z", np.deg2rad(10.0), steps, 3)
    timeout = 10./steps

    for i, to in enumerate(traj):
        cmd = np.concatenate([initial_pose[:3], to])
        # print("Initial error", i, np.round(spalg.translation_rotation_error(traj[0], arm.end_effector())[3:], 4))
        arm.set_target_pose_flex(cmd, t=timeout)
        rospy.sleep(timeout)


def execute_trajectory(trajectory, duration, use_force_control=False, termination_criteria=None):
    if use_force_control:
        pf_model = init_force_control([0., 0.8, 0.8, 0.8, 0.8, 0.8])
        # pf_model = init_force_control([0., 1., 1., 1., 1., 1.])
        target_force = np.array([0., 0., 0., 0., 0., 0.])
        max_force_torque = np.array([50., 50., 50., 5., 5., 5.])

        def termination_criteria(current_pose): return current_pose[0] > 1

        full_force_control(target_force, trajectory, pf_model, timeout=duration,
                        relative_to_ee=False, max_force_torque=max_force_torque, termination_criteria=termination_criteria)

    else:
        joint_trajectory = []
        for point in trajectory:
            joint_trajectory.append(arm._solve_ik(point))
        arm.set_joint_trajectory(joint_trajectory, t=duration)

def face_towards_target():
    """
        Move robot's end-effector towards a target point. 
    """
    cpose = arm.end_effector()  # current pose
    target_position = [0.35951, -0.54521, 0.34393]
    # compute pose with new rotation
    cmd = spalg.face_towards(target_position, cpose)
    arm.set_target_pose(cmd, wait=True, t=1)


def init_force_control(selection_matrix, dt=0.002):
    Kp = np.array([1., 1., 1., 2.5, 2.5, 2.5])
    Kp_pos = Kp
    Kd_pos = Kp * 0.01
    Ki_pos = Kp * 0.01
    position_pd = utils.PID(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, dynamic_pid=True)

    # Force PID gains
    Kp = np.array([0.05, 0.05, 0.05, 5.0, 5.0, 5.0])
    Kp_force = Kp
    Kd_force = Kp * 0.01
    Ki_force = Kp * 0.01
    force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
    pf_model = ForcePositionController(position_pd=position_pd, force_pd=force_pd, alpha=np.diag(selection_matrix), dt=dt)

    return pf_model


def full_force_control(
        target_force=None, target_positions=None, model=None,
        selection_matrix=[1., 1., 1., 1., 1., 1.], 
        relative_to_ee=False, timeout=10.0, max_force_torque=[50., 50., 50., 5., 5., 5.],
        termination_criteria=None):
    """ 
      Use with caution!! 
      target_force: list[6], target force for each direction x,y,z,ax,ay,az
      target_position: list[7], target position for each direction x,y,z + quaternion
      selection_matrix: list[6], define which direction is controlled by position(1.0) or force(0.0)
      relative_to_ee: bool, whether to use the base_link of the robot as frame or the ee_link (+ ee_transform)
      timeout: float, duration in seconds of the force control
    """
    arm.set_wrench_offset(True)  # offset the force sensor
    arm.relative_to_ee = relative_to_ee

    # TODO(cambel): Define a config file for the force-control parameters
    if model is None:
        pf_model = init_force_control(selection_matrix)
    else:
        pf_model = model
        pf_model.selection_matrix = np.diag(selection_matrix)

    max_force_torque = np.array(max_force_torque)

    target_force = np.array([0., 0., 0., 0., 0., 0.]) if target_force is None else target_force

    target_positions = arm.end_effector() if target_positions is None else np.array(target_positions)

    pf_model.set_goals(force=target_force)

    # print("STARTING Force Control with target_force:", target_force, "timeout", timeout)
    return arm.set_hybrid_control_trajectory(target_positions, pf_model, max_force_torque=max_force_torque, timeout=timeout, stop_on_target_force=False, termination_criteria=termination_criteria)
    # rospy.loginfo("Force control finished with: %s" % res)  # debug


def force_control():
    arm.set_wrench_offset(True)

    timeout = 5.0

    selection_matrix = [0., 1., 1., 1., 1., 1.]
    target_force = np.array([-5., 0., 0., 0., 0., 0.])

    full_force_control(target_force, selection_matrix=selection_matrix, timeout=timeout, relative_to_ee=False)


def execute_manual_routine(routine_filename):
    path = rospkg.RosPack().get_path("o2ac_routines") + ("/config/%s.yaml" % routine_filename)
    with open(path, 'r') as f:
        routine = yaml.load(f)
    robot_name = routine["robot_name"]
    waypoints = routine["waypoints"]

    for i, point in enumerate(waypoints):
        print("point:", i+1)
        raw_input()
        pose = point['pose']
        pose_type = point['type']
        gripper_action = point.get('gripper-action')
        duration = point['duration']
        move_to_waypoint(pose, pose_type, gripper_action, 1.)

def move_to_waypoint(pose, pose_type, gripper_action, duration):
    if pose_type == 'joint-space':
        target=pose = arm.end_effector(pose)
        # arm.set_joint_positions(pose, wait=True, t=1.0)
        arm.move_linear(target, t=duration)
    elif pose_type == 'task-space':
        arm.set_target_pose(pose, wait=True, t=duration)
    elif pose_type == 'relative-tcp':
        arm.move_relative(pose, relative_to_ee=True, t=duration)
    elif pose_type == 'relative-base':
        arm.move_relative(pose, relative_to_ee=False, t=duration)
    else:
        raise ValueError("Invalid pose_type: %s" % pose_type)
    if gripper_action:
        pass # do gripper action


def move_linear():
    # get current position of the end effector
    cpose = arm.end_effector()
    # define the desired translation/rotation
    deltax = np.array([0.1, 0.0, 0.0, 0., np.deg2rad(15.), 0.])
    # add translation/rotation to current position
    cmd = transformations.pose_euler_to_quaternion(cpose, deltax)
    print(cmd[3:], cpose[3:])
    # execute desired new pose
    # may fail if IK solution is not found
    arm.move_linear(pose=cmd, t=1.0)

    cpose = arm.end_effector()
    deltax = np.array([0.0, 0.1, 0.0, 0., np.deg2rad(-15.), 0.])
    # add translation/rotation to current position
    cmd = transformations.pose_euler_to_quaternion(cpose, deltax)
    # execute desired new pose
    # may fail if IK solution is not found
    arm.move_linear(pose=cmd, t=1.0)

    cpose = arm.end_effector()
    deltax = np.array([-0.1, 0.0, 0.0, np.deg2rad(15.), 0., 0.])
    # add translation/rotation to current position
    cmd = transformations.pose_euler_to_quaternion(cpose, deltax)
    # execute desired new pose
    # may fail if IK solution is not found
    arm.move_linear(pose=cmd, t=1.0)

    cpose = arm.end_effector()
    deltax = np.array([0.0, -0.1, 0.0, np.deg2rad(-15.), 0., 0.])
    # add translation/rotation to current position
    cmd = transformations.pose_euler_to_quaternion(cpose, deltax)
    # execute desired new pose
    # may fail if IK solution is not found
    arm.move_linear(pose=cmd, t=1.0)

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
    parser.add_argument('-p', '--pose', action='store_true',
                        help='Move to pose')
    parser.add_argument('-w', '--wiggle', action='store_true',
                        help='Wiggle')
    parser.add_argument('-r', '--routine', action='store_true',
                        help='Execute manual routine')
    parser.add_argument('--grasp_naive', action='store_true',
                        help='Test simple grasping (cube_tasks world)')
    parser.add_argument('--grasp_plugin', action='store_true',
                        help='Test grasping plugin (cube_tasks world)')
    parser.add_argument('--circle', action='store_true',
                        help='Circular rotation around a target pose')
    parser.add_argument('--spiral', action='store_true',
                        help='Spiral rotation around a target pose')
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
    rospackage = None
    tcp_link = None
    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
        rospackage = "o2ac_scene_description"
        tcp_link='tool0'

    use_gripper = args.gripper

    extra_ee = [0, 0, 0.] + transformations.quaternion_from_euler(*[np.pi/4, 0, 0]).tolist()
    extra_ee = [0.0, 0.0, 0.173, 0., 0., 0., 1.]
    extra_ee = [0.0, 0.0, 0.173, 0.500, -0.500, 0.500, 0.500]

    global arm
    arm = CompliantController(ft_sensor=True, ee_transform=extra_ee,
                              gripper=use_gripper, namespace=ns,
                              joint_names_prefix=joints_prefix,
                              robot_urdf=robot_urdf, robot_urdf_package=rospackage, 
                              ee_link=tcp_link)
    print("Extra ee", extra_ee)

    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    if args.move:
        move_joints()
    if args.pose:
        move_linear()
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
    if args.spiral:
        spiral_trajectory()
    if args.face:
        face_towards_target()
    if args.force:
        force_control()
    if args.wiggle:
        wiggle()
    if args.routine:
        execute_manual_routine("bearing_orient_totb")
    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
