# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rospy
import numpy as np
import types
import timeit

from ur_control.arm import Arm
from ur_control import transformations, spalg, utils
from ur_control.constants import DONE, FORCE_TORQUE_EXCEEDED, SPEED_LIMIT_EXCEEDED, STOP_ON_TARGET_FORCE, TERMINATION_CRITERIA, IK_NOT_FOUND


class CompliantController(Arm):
    def __init__(self,
                 relative_to_ee=False,
                 **kwargs):
        """ Compliant controller
            relative_to_ee bool: if True when moving in task-space move relative to the end-effector otherwise
                            move relative to the world coordinates
        """
        Arm.__init__(self, **kwargs)

        self.relative_to_ee = relative_to_ee

        # read publish rate if it does exist, otherwise set publish rate
        js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 500.0)
        self.rate = rospy.Rate(js_rate)

    def set_hybrid_control_trajectory(self, trajectory, model, max_force_torque, timeout=5.0, 
                                            stop_on_target_force=False, termination_criteria=None,
                                            displacement_epsilon=0.001, verbose=False):
        """ Move the robot according to a hybrid controller model
            trajectory: array[array[7]] or array[7], can define a single target pose or a trajectory of multiple poses.
            model: force control model, see hybrid_controller.py 
            max_force_torque: array[6], max force/torque allowed before stopping controller
            timeout: float, maximum duration of controller's operation
            stop_on_target_force: bool: stop once the model's target force has been achieved, stopping controller when all non-zero target forces/torques are reached
            termination_criteria: lambda/function, special termination criteria based on current pose of the robot w.r.t the robot's base
            displacement_epsilon: float,  if there is no motion larger than this, then start counting the standby time
        """

        reduced_speed = np.deg2rad([50, 50, 50, 100, 100, 100])

        xb = self.end_effector()
        failure_counter = 0

        ptp_index = 0
        q_last = self.joint_angles()

        if trajectory.ndim == 1:  # just one point
            ptp_timeout = timeout
            model.set_goals(position=trajectory)
        else:  # trajectory
            ptp_timeout = timeout / len(trajectory)
            model.set_goals(position=trajectory[ptp_index])

        log = {SPEED_LIMIT_EXCEEDED:0, IK_NOT_FOUND:0}

        result = DONE

        standby_timer = timeit.default_timer()
        displacement_dt = 0.0 # overall euclidean distance

        # Timeout for motion
        initime = rospy.get_time()
        sub_inittime = rospy.get_time()
        while not rospy.is_shutdown() \
                and (rospy.get_time() - initime) < timeout:

            # Transform wrench to the base_link frame
            Wb = self.get_ee_wrench()
            # Current position in task-space
            xb = self.end_effector()

            if termination_criteria is not None:
                assert isinstance(termination_criteria, types.LambdaType), "Invalid termination criteria, expecting lambda/function with one argument[current pose array[7]]"
                standby_time = timeit.default_timer() - standby_timer
                if termination_criteria(xb, standby_time):
                    rospy.loginfo("Termination criteria returned True, stopping force control")
                    result = TERMINATION_CRITERIA
                    break

            if (rospy.get_time() - sub_inittime) > ptp_timeout:
                sub_inittime = rospy.get_time()
                ptp_index += 1
                if ptp_index >= len(trajectory):
                    rospy.loginfo("Trajectory completed")
                    result = DONE
                    break
                if not trajectory.ndim == 1: # For some reason the timeout validation is not robust enough
                    model.set_goals(position=trajectory[ptp_index])

            Fb = -1 * Wb
            if stop_on_target_force and np.all(np.abs(Fb)[model.target_force != 0] > np.abs(model.target_force)[model.target_force != 0]):
                rospy.loginfo('Target F/T reached {}'.format(np.round(Wb, 3)) + ' Stopping!')
                self.set_target_pose_flex(pose=xb, t=model.dt)
                result = STOP_ON_TARGET_FORCE
                break

            # Safety limits: max force
            if np.any(np.abs(Wb) > max_force_torque):
                rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(Wb, 3)))
                self.set_target_pose_flex(pose=xb, t=model.dt)
                result = FORCE_TORQUE_EXCEEDED
                break

            # Current Force in task-space
            dxf = model.control_position_orientation(Fb, xb)  # angular velocity

            # Limit linear/angular velocity
            dxf[:3] = np.clip(dxf[:3], -0.5, 0.5)
            dxf[3:] = np.clip(dxf[3:], -5., 5.)

            xc = transformations.pose_from_angular_veloticy(xb, dxf, dt=model.dt)

            # Avoid extra acceleration when a point failed due to IK or other violation
            # So, this corrects the allowed time for the next point
            dt = model.dt * (failure_counter+1) 
            
            q = self._solve_ik(xc)
            if q is None:
                rospy.logwarn("IK not found")
                result = IK_NOT_FOUND
            else:
                q_speed = (q_last - q)/dt
                if np.any(np.abs(q_speed) > reduced_speed):
                    rospy.logwarn_once("Exceeded reduced max speed %s deg/s, Ignoring command" % np.round(np.rad2deg(q_speed),0)) 
                    result = SPEED_LIMIT_EXCEEDED
                else:
                    result = self.set_joint_positions_flex(position=q, t=dt)

            if result != DONE:
                failure_counter += 1
                if result == IK_NOT_FOUND:
                    log[IK_NOT_FOUND] += 1
                if result == SPEED_LIMIT_EXCEEDED:
                    log[SPEED_LIMIT_EXCEEDED] += 1
                continue # Don't wait since there is not motion
            else:
                failure_counter = 0

            # Compensate the time allocated to the next command when there are failures
            # Especially important for following a motion trajectory
            for _ in range(failure_counter+1):
                self.rate.sleep()

            displacement_dt += np.linalg.norm(self.end_effector(q_last)[:3] - self.end_effector()[:3])
            if displacement_dt > displacement_epsilon:
                standby_timer = timeit.default_timer() # restart timer
                displacement_dt = 0.0

            q_last = self.joint_angles()
        
        if verbose:
            rospy.logwarn("Total # of commands ignored: %s" % log)
        return result

    # TODO(cambel): organize this code to avoid this repetition of code
    def set_hybrid_control(self, model, max_force_torque, timeout=5.0, stop_on_target_force=False):
        """ Move the robot according to a hybrid controller model"""
        
        reduced_speed = np.deg2rad([100, 100, 100, 150, 150, 150])
        q_last = self.joint_angles()

        # Timeout for motion
        initime = rospy.get_time()
        xb = self.end_effector()
        failure_counter = 0

        while not rospy.is_shutdown() \
                and (rospy.get_time() - initime) < timeout:

            # Transform wrench to the base_link frame
            Wb = self.get_ee_wrench()

            # Current Force in task-space
            Fb = -1 * Wb
            # Safety limits: max force
            if np.any(np.abs(Fb) > max_force_torque):
                rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(Wb, 3)))
                self.set_target_pose_flex(pose=xb, t=model.dt)
                return FORCE_TORQUE_EXCEEDED

            if stop_on_target_force and np.any(np.abs(Fb)[model.target_force != 0] > model.target_force[model.target_force != 0]):
                rospy.loginfo('Target F/T reached {}'.format(np.round(Wb, 3)) + ' Stopping!')
                self.set_target_pose_flex(pose=xb, t=model.dt)
                return STOP_ON_TARGET_FORCE

            # Current position in task-space
            xb = self.end_effector()

            dxf = model.control_position_orientation(Fb, xb)  # angular velocity
            
            # Limit linear/angular velocity
            dxf[:3] = np.clip(dxf[:3], -0.5, 0.5)
            dxf[3:] = np.clip(dxf[3:], -5., 5.)
            
            xc = transformations.pose_from_angular_veloticy(xb, dxf, dt=model.dt)

            # Avoid extra acceleration when a point failed due to IK or other violation
            # So, this corrects the allowed time for the next point
            dt = model.dt * (failure_counter+1) 

            q = self._solve_ik(xc)
            if q is None:
                rospy.logwarn("IK not found")
                result = IK_NOT_FOUND
            else:
                q_speed = (q_last - q)/dt
                if np.any(np.abs(q_speed) > reduced_speed):
                    rospy.logwarn("Exceeded reduced max speed %s deg/s, Ignoring command" % np.round(np.rad2deg(q_speed),0)) 
                    result = SPEED_LIMIT_EXCEEDED
                else:
                    result = self.set_joint_positions_flex(position=q, t=dt)
            
            if result != DONE:
                failure_counter += 1
                continue # Don't wait since there is not motion
            else:
                failure_counter = 0

            # Compensate the time allocated to the next command when there are failures
            for _ in range(failure_counter+1):
                self.rate.sleep()

            q_last = self.joint_angles()
        return DONE
