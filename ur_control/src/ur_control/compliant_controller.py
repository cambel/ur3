# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
 
import rospy
import numpy as np

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
                rospy.loginfo('Target force/torque reached {}'.format(np.round(Wb, 3)) + ' Stopping!')
                self.set_target_pose_flex(pose=xb, t=model.dt)
                return STOP_ON_TARGET_FORCE

            # Current position in task-space
            xb = self.end_effector()

            dxf = model.control_position_orientation(Fb, xb)  # angular velocity
            # TODO (cambel): smooth motion (velocity/acceleration)
            xc = transformations.pose_from_angular_veloticy(xb, dxf, dt=model.dt)

            result = self.set_target_pose_flex(pose=xc, t=model.dt)
            # if result != DONE:
            #     return result

            self.rate.sleep()
        return DONE
