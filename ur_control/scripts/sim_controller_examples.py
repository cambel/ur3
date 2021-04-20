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
    q = [1.5701, -1.1854, 1.3136, -1.6975, -1.5708, -0.0016]
    q = [1.5794, -1.4553, 2.1418, -2.8737, -1.6081, 0.0063]
    q = [1.7078, -1.5267, 2.0624, -2.1325, -1.6114, 1.7185] #b_bot
    q = [1.5909, -1.3506, 1.9397, -2.0634, -2.5136, -1.4549] #b_bot bearing
    # q = [1.707, -1.5101, 2.1833, -2.5707, -1.6139, 1.7138]
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

def move_to_pose():
    cpose = arm.end_effector()
    cpose[3:] = [0,0,0,1]
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

def compute_trajectory(initial_q, deltax, steps, revolutions=1.0, traj_type="circular"):
    arm.set_joint_positions(initial_q, wait=True, t=2)

    initial_pose = arm.end_effector()

    target_pose = transformations.pose_euler_to_quaternion(initial_pose, deltax, ee_rotation=True)

    initial_pose = initial_pose[:3]
    final_pose = target_pose[:3]

    target_q = transformations.vector_to_pyquaternion(target_pose[3:])
    target_q = transformations.vector_to_pyquaternion(transformations.quaternion_from_euler(*[0, np.pi/2, 0]))

    p1 = np.zeros(3)
    p2 = target_q.rotate(initial_pose - final_pose)

    if traj_type == "circular":
        traj = traj_utils.get_circular_trajectory(p1, p2, steps, revolutions)
    if traj_type == "spiral":
        traj = traj_utils.get_spiral_trajectory(p1, p2, steps, revolutions, from_center=True)
    
    traj = np.apply_along_axis(target_q.rotate, 1, traj)
    trajectory = traj + final_pose

    trajectory = [np.concatenate([t, target_pose[3:]]) for t in trajectory]

    return trajectory

def spiral_trajectory():
    initial_q = [1.5909, -1.3506, 1.9397, -2.0634, -2.5136, -1.4549] #b_bot bearing
    deltax = np.array([0.002, 0.0, 0.0, 0., 0., 0.])

    steps = 200
    duration = 10.0
    
    trajectory = compute_trajectory(initial_q, deltax, steps, revolutions=5, traj_type="spiral")
    execute_trajectory(trajectory, timeout=(duration/steps), use_force_control=True)

def circular_trajectory():
    initial_q = [1.5909, -1.3506, 1.9397, -2.0634, -2.5136, -1.4549] #b_bot bearing
    deltax = np.array([0.002, 0.0, 0.0, 0., 0., 0.])

    steps = 200
    duration = 10.0

    trajectory = compute_trajectory(initial_q, deltax, steps, revolutions=5, traj_type="circular")
    execute_trajectory(trajectory, timeout=(duration/steps), use_force_control=True)

def execute_trajectory(trajectory, timeout, use_force_control=False):
    if use_force_control:
        pf_model = init_force_control([0.,1.,1.,1.,1.,1.])
        ee_tranform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        target_force = np.array([-1., 0., 0., 0., 0., 0.])
        arm.set_wrench_offset(True)

    arm.set_target_pose(trajectory[0], wait=True, t=2)

    for cmd in trajectory:
        if use_force_control:
            full_force_control(target_force, cmd, pf_model, ee_transform=ee_tranform, timeout=timeout, relative_to_ee=False)
        else:
            arm.set_target_pose_flex(cmd, t=timeout)
            rospy.sleep(timeout)
            
    print("relative error", np.round(spalg.translation_rotation_error(trajectory[-1], arm.end_effector()), 4))


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
    Kp = np.array([1., 1., 1., 10., 10., 10.])
    Kp_pos = Kp
    Kd_pos = Kp * 0.01
    Ki_pos = Kp * 0.01
    position_pd = utils.PID(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, dynamic_pid=True)

    # Force PID gains
    Kp = np.array([0.02, 0.05, 0.05, 0.5, 0.5, 5.0])
    Kp_force = Kp
    Kd_force = Kp * 0.
    Ki_force = Kp * 0.01
    force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
    pf_model = ForcePositionController(position_pd=position_pd, force_pd=force_pd, alpha=np.diag(selection_matrix), dt=dt)

    return pf_model

def full_force_control(target_force=None, target_position=None, model=None, selection_matrix=[1., 1., 1., 1., 1., 1.], ee_transform=[0, 0, 0, 0, 0, 0, 1], relative_to_ee=False, timeout=10.0,):
    """ 
      Use with caution!! 
      target_force: list[6], target force for each direction x,y,z,ax,ay,az
      target_position: list[7], target position for each direction x,y,z + quaternion
      selection_matrix: list[6], define which direction is controlled by position(1.0) or force(0.0)
      ee_transform: list[7], additional transformation of the end-effector (e.g to match tool or special orientation) x,y,z + quaternion
      relative_to_ee: bool, whether to use the base_link of the robot as frame or the ee_link (+ ee_transform)
      timeout: float, duration in seconds of the force control
    """
    arm.set_wrench_offset(True)  # offset the force sensor
    arm.relative_to_ee = relative_to_ee
    arm.ee_transform = ee_transform

    # TODO(cambel): Define a config file for the force-control parameters
    if model is None:
        pf_model = init_force_control(selection_matrix)
    else:
        pf_model = model
        pf_model.selection_matrix = np.diag(selection_matrix)

    max_force_torque = np.array([50., 50., 50., 5., 5., 5.])

    target_position = arm.end_effector() if target_position is None else np.array(target_position)
    target_force = np.array([0., 0., 0., 0., 0., 0.]) if target_force is None else target_force

    pf_model.set_goals(target_position, target_force)

    # print("STARTING Force Control with target_force:", target_force, "timeout", timeout)
    res = arm.set_hybrid_control(pf_model, max_force_torque=max_force_torque, timeout=timeout, stop_on_target_force=False)
    # rospy.loginfo("Force control finished with: %s" % res)  # debug


def force_control():
    arm.set_wrench_offset(True)

    timeout = 30.0

    selection_matrix = [0.5, 0.9, 0.9, 1., 1., 1.]
    target_position = arm.end_effector()
    # target_position[1] += 0.05
    target_force = np.array([0., 0., 0., 0., 0., 0.])
    # ee_tranform = [0, 0, 0.] + transformations.quaternion_from_euler(*[np.pi/2, 0, 0]).tolist()
    ee_tranform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # print(ee_tranform)
    full_force_control(target_force, target_position, selection_matrix=selection_matrix, ee_transform=ee_tranform, timeout=timeout, relative_to_ee=False)


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
    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + "_"
        robot_urdf = args.namespace
        rospackage = "o2ac_scene_description"

    use_gripper = args.gripper

    extra_ee = [0,0,0.] + transformations.quaternion_from_euler(*[np.pi/4,0,0]).tolist()
    extra_ee = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    global arm
    arm = CompliantController(ft_sensor=True, ee_transform=extra_ee,
                              gripper=use_gripper, namespace=ns,
                              joint_names_prefix=joints_prefix,
                              robot_urdf=robot_urdf, robot_urdf_package=rospackage)
    print("Extra ee", extra_ee)

    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    if args.move:
        move_joints()
    if args.pose:
        move_to_pose()
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

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
