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

from ur_control.arm import Arm
from ur_control import transformations, spalg, utils
from ur_control.constants import DONE, FORCE_TORQUE_EXCEEDED, SPEED_LIMIT_EXCEEDED, STOP_ON_TARGET_FORCE


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

    def set_hybrid_control_trajectory(self, trajectory, model, max_force_torque, timeout=5.0, stop_on_target_force=False, termination_criteria=None):
        """ Move the robot according to a hybrid controller model
            trajectory: array[array[7]] or array[7], can define a single target pose or a trajectory of multiple poses.
            model: force control model, see hybrid_controller.py 
            max_force_torque: array[6], max force/torque allowed before stopping controller
            timeout: float, maximum duration of controller's operation
            stop_on_target_force: bool: stop once the model's target force has been achieved, stopping controller when all non-zero target forces/torques are reached
            termination_criteria: lambda/function, special termination criteria based on current pose of the robot w.r.t the robot's base
        """

        xb = self.end_effector()

        ptp_index = 0

        if trajectory.ndim == 1:  # just one point
            ptp_timeout = timeout
            model.set_goals(position=trajectory)
        else:  # trajectory
            ptp_timeout = timeout / len(trajectory)
            model.set_goals(position=trajectory[ptp_index])

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
                if termination_criteria(xb):
                    rospy.loginfo("Termination criteria returned True, stopping force control")
                    break

            if (rospy.get_time() - sub_inittime) > ptp_timeout:
                sub_inittime = rospy.get_time()
                ptp_index += 1
                if ptp_index >= len(trajectory):
                    rospy.loginfo("Trajectory completed")
                    break
                model.set_goals(position=trajectory[ptp_index])

            if stop_on_target_force and np.all(np.abs(Wb)[model.target_force != 0] > model.target_force[model.target_force != 0]):
                rospy.loginfo('Target F/T reached {}'.format(np.round(Wb, 3)) + ' Stopping!')
                self.set_target_pose_flex(pose=xb, t=model.dt)
                return STOP_ON_TARGET_FORCE

            # Safety limits: max force
            if np.any(np.abs(Wb) > max_force_torque):
                rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(Wb, 3)))
                self.set_target_pose_flex(pose=xb, t=model.dt)
                return FORCE_TORQUE_EXCEEDED

            # Current Force in task-space
            Fb = -1 * Wb
            dxf = model.control_position_orientation(Fb, xb)  # angular velocity

            # Limit linear/angular velocity
            dxf[:3] = np.clip(dxf[:3], -1., 1.)
            dxf[3:] = np.clip(dxf[3:], -5., 5.)

            xc = transformations.pose_from_angular_veloticy(xb, dxf, dt=model.dt)

            self.set_target_pose_flex(pose=xc, t=model.dt)

            self.rate.sleep()
        return DONE

    def set_hybrid_control(self, model, max_force_torque, timeout=5.0, stop_on_target_force=False):
        """ Move the robot according to a hybrid controller model"""
        # Timeout for motion
        initime = rospy.get_time()
        xb = self.end_effector()
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
            xc = transformations.pose_from_angular_veloticy(xb, dxf, dt=model.dt)

            result = self.set_target_pose_flex(pose=xc, t=model.dt)
            # if result != DONE:
            #     return result

            self.rate.sleep()
        return DONE
