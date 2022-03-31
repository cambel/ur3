#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import sys
import signal
from ur_control import utils, traj_utils
from ur_control.hybrid_controller import ForcePositionController
from ur_control.compliant_controller import CompliantController
import argparse
import rospy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def move_joints(wait=True):
    # desired joint configuration 'q'
    q = [1.57, -1.57, 1.26, -1.57, -1.57, 0]

    # go to desired joint configuration
    # in t time (seconds)
    # wait is for waiting to finish the motion before executing
    # anything else or ignore and continue with whatever is next
    arm.set_joint_positions(position=q, wait=wait, t=2.0)


def spiral_trajectory():
    """
        Force/Position control. Follow a spiral trajectory on the world's YZ plan while controlling force on Z 
    """
    initial_q = [1.57, -1.57, 1.26, -1.57, -1.57, 0]

    arm.set_joint_positions(initial_q, wait=True, t=2)

    plane = "YZ"
    radius = 0.02
    radius_direction = "+Z"
    revolutions = 3

    steps = 100 # Number of waypoints of the spiral trajectory
    duration = 30.0 # Duration of the trajectory, affects speed

    arm.set_wrench_offset(True)

    initial_pose = arm.end_effector()
    trajectory = traj_utils.compute_trajectory(initial_pose, plane, radius, radius_direction,
                                               steps, revolutions, trajectory_type="spiral", from_center=True,
                                               wiggle_direction="X", wiggle_angle=np.deg2rad(0.0), wiggle_revolutions=1.0)
    execute_trajectory(trajectory, duration=duration, use_force_control=True)


def circular_trajectory():
    """
        Force/Position control. Follow a circular trajectory on the world's YZ plan while controlling force on Z 
    """
    initial_q = [1.57, -1.57, 1.26, -1.57, -1.57, 0]
    
    arm.set_joint_positions(initial_q, wait=True, t=1)

    plane = "YZ"
    radius = 0.02
    radius_direction = "+Z"
    revolutions = 1

    steps = 100 # Number of waypoints of the circular trajectory
    duration = 30.0 # Duration of the trajectory, affects speed

    arm.set_wrench_offset(True)

    initial_pose = arm.end_effector()
    trajectory = traj_utils.compute_trajectory(initial_pose, plane, radius, radius_direction,
                                               steps, revolutions, trajectory_type="circular", from_center=False,
                                               wiggle_direction="X", wiggle_angle=np.deg2rad(0.0), wiggle_revolutions=10.0)
    execute_trajectory(trajectory, duration=duration, use_force_control=True)


def execute_trajectory(trajectory, duration, use_force_control=False, termination_criteria=None):
    if use_force_control:
        pf_model = init_force_control([1., 1., 1., 1., 1., 1.])
        target_force = np.array([0., 0., 0., 0., 0., 0.])
        max_force_torque = np.array([50.0, 50., 50., 5., 5., 5.])

        def termination_criteria(current_pose, standby): return False # Dummy function

        full_force_control(target_force, trajectory, pf_model, timeout=duration,
                           relative_to_ee=False, max_force_torque=max_force_torque, termination_criteria=termination_criteria)

    else:
        joint_trajectory = []
        for point in trajectory:
            joint_trajectory.append(arm._solve_ik(point))
        arm.set_joint_trajectory(joint_trajectory, t=duration)


def init_force_control(selection_matrix, dt=0.002):
    Kp = np.array([3., 3., 3., 1., 1., 1.])
    Kp_pos = Kp
    Kd_pos = Kp * 0.01
    Ki_pos = Kp * 0.01
    position_pd = utils.PID(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, dynamic_pid=True)

    # Force PID gains
    Kp = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    Kp_force = Kp
    Kd_force = Kp * 0.01
    Ki_force = Kp * 0.01
    force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
    pf_model = ForcePositionController(
        position_pd=position_pd, force_pd=force_pd, alpha=np.diag(selection_matrix), dt=dt)

    return pf_model


def full_force_control(
        target_force=None, target_positions=None, model=None,
        selection_matrix=[1., 1., 1., 1., 1., 1.],
        relative_to_ee=False, timeout=10.0, max_force_torque=[200., 200., 200., 5., 5., 5.],
        termination_criteria=None):
    """ 
      Use with caution!! 
      target_force: list[6], target force for each direction x,y,z,ax,ay,az
      target_position: list[7], target position for each direction x,y,z + quaternion
      selection_matrix: list[6], define which direction is controlled by position(1.0) or force(0.0)
      relative_to_ee: bool, whether to use the base_link of the robot as frame or the ee_link (+ ee_transform)
      timeout: float, duration in seconds of the force control
      termination_criteria: func, optional condition that would stop the compliance controller
    """
    arm.set_wrench_offset(True)  # offset the force sensor
    arm.relative_to_ee = relative_to_ee

    if model is None:
        pf_model = init_force_control(selection_matrix)
    else:
        pf_model = model
        pf_model.selection_matrix = np.diag(selection_matrix)

    max_force_torque = np.array(max_force_torque)

    target_force = np.array([0., 0., 0., 0., 0., 0.]
                            ) if target_force is None else target_force

    target_positions = arm.end_effector(
    ) if target_positions is None else np.array(target_positions)

    pf_model.set_goals(force=target_force)

    return arm.set_hybrid_control_trajectory(target_positions, pf_model, max_force_torque=max_force_torque,
                                             timeout=timeout, stop_on_target_force=False,
                                             termination_criteria=termination_criteria)

def force_control():
    """ 
        Simple example of compliance control
        selection_matrix: list[6]. define which direction is controlled by position(1.0) or force(0.0) goal. 
                          Values in between make the controller attempt to achieve both position and force goals.
    """
    arm.set_wrench_offset(True)

    timeout = 10.0  # Duration of the active control, does not affect speed.

    selection_matrix = [1., 1., 0., 1., 1., 1.]
    target_force = np.array([0., 0., 1., 0., 0., 0.])

    full_force_control(
        target_force, selection_matrix=selection_matrix, timeout=timeout)


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force control demo')
    parser.add_argument('--circle', action='store_true',
                        help='Circular rotation around a target pose')
    parser.add_argument('--spiral', action='store_true',
                        help='Spiral rotation around a target pose')
    parser.add_argument('--namespace', type=str, 
                        help='Namespace of arm', default=None)
    args = parser.parse_args()

    rospy.init_node('ur3e_compliance_control')

    ns = ''
    joints_prefix = None
    robot_urdf = "ur3e"
    tcp_link = None

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + '_'

    global arm
    arm = CompliantController(ft_sensor=True,
                              namespace=ns,
                              joint_names_prefix=joints_prefix,
                              robot_urdf=robot_urdf,
                              ee_link=tcp_link)

    if args.move:
        move_joints()
    if args.circle:
        circular_trajectory()
    if args.spiral:
        spiral_trajectory()
    if args.force:
        force_control()


if __name__ == "__main__":
    main()
