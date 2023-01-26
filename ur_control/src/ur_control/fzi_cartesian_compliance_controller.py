# Copyright (c) 2018-2021, Cristian Beltran.  All rights reserved.
#
# Cristian Beltran and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from Cristian Beltran is strictly prohibited.

import rospy
import numpy as np

from ur_control.arm import Arm
from ur_control import transformations, utils, conversions
from ur_control.constants import JOINT_TRAJECTORY_CONTROLLER, CARTESIAN_COMPLIANCE_CONTROLLER, STOP_ON_TARGET_FORCE, FORCE_TORQUE_EXCEEDED, DONE

from geometry_msgs.msg import WrenchStamped, PoseStamped

import dynamic_reconfigure.client


def convert_selection_matrix_to_parameters(selection_matrix):
    return {
        "stiffness":
        {
            "sel_x": selection_matrix[0],
            "sel_y": selection_matrix[1],
            "sel_z": selection_matrix[2],
            "sel_ax": selection_matrix[3],
            "sel_ay": selection_matrix[4],
            "sel_az": selection_matrix[5],
        }
    }


def convert_pd_gains_to_parameters(p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
    return {
        "trans_x": {"p": p_gains[0], "d": d_gains[0]},
        "trans_y": {"p": p_gains[1], "d": d_gains[1]},
        "trans_z": {"p": p_gains[2], "d": d_gains[2]},
        "rot_x": {"p": p_gains[3], "d": d_gains[3]},
        "rot_y": {"p": p_gains[4], "d": d_gains[4]},
        "rot_z": {"p": p_gains[5], "d": d_gains[5]}
    }


def switch_cartesian_controllers(func):
    '''Decorator that switches from cartesian to joint trajectory controllers and back'''

    def wrap(*args, **kwargs):
        args[0].controller_manager.switch_controllers(controllers_on=[CARTESIAN_COMPLIANCE_CONTROLLER],
                                                      controllers_off=[JOINT_TRAJECTORY_CONTROLLER])
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            print(e)
            res = DONE

        args[0].controller_manager.switch_controllers(controllers_on=[JOINT_TRAJECTORY_CONTROLLER],
                                                      controllers_off=[CARTESIAN_COMPLIANCE_CONTROLLER])
        return res
    return wrap


class CompliantController(Arm):
    def __init__(self,
                 **kwargs):
        """ Compliant controller using FZI Cartesian Compliance controllers """
        Arm.__init__(self, **kwargs)
        self.cartesian_target_pose_pub = rospy.Publisher('%s/%s/target_frame' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), PoseStamped, queue_size=10.0)
        self.cartesian_target_wrench_pub = rospy.Publisher('%s/%s/target_wrench' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), WrenchStamped, queue_size=10.0)

        self.dyn_config_clients = {
            "trans_x": dynamic_reconfigure.client.Client("%s/%s/pd_gains/trans_x" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "trans_y": dynamic_reconfigure.client.Client("%s/%s/pd_gains/trans_y" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "trans_z": dynamic_reconfigure.client.Client("%s/%s/pd_gains/trans_z" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_x": dynamic_reconfigure.client.Client("%s/%s/pd_gains/rot_x" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_y": dynamic_reconfigure.client.Client("%s/%s/pd_gains/rot_y" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_z": dynamic_reconfigure.client.Client("%s/%s/pd_gains/rot_z" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "stiffness": dynamic_reconfigure.client.Client("%s/%s/stiffness" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "hand_frame_control": dynamic_reconfigure.client.Client("%s/%s" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
        }
        rospy.on_shutdown(self.safety_hook)
        self.set_hand_frame_control(False)

    def safety_hook(self):
        self.controller_manager.switch_controllers(controllers_on=[JOINT_TRAJECTORY_CONTROLLER],
                                                   controllers_off=[CARTESIAN_COMPLIANCE_CONTROLLER])

    def set_cartesian_target_wrench(self, wrench: list):
        # Publish the target wrench
        try:
            target_wrench = WrenchStamped()
            target_wrench.header.frame_id = self.base_link
            target_wrench.wrench = conversions.to_wrench(wrench)
            self.cartesian_target_wrench_pub.publish(target_wrench)
        except Exception as e:
            rospy.logerr("Fail to set_target_wrench(): %s" % e)

    def set_cartesian_target_pose(self, pose: list):
        # Publish the target pose
        try:
            target_pose = conversions.to_pose_stamped(self.base_link, pose)
            self.cartesian_target_pose_pub.publish(target_pose)
        except Exception as e:
            rospy.logerr("Fail to set_target_pose(): %s" % e)

    def update_controller_parameters(self, parameters: dict):
        for param in parameters.keys():
            rospy.logdebug("Setting parameters %s to the group %s" % (parameters[param], param))
            self.dyn_config_clients[param].update_configuration(parameters[param])

    def update_selection_matrix(self, selection_matrix):
        parameters = convert_selection_matrix_to_parameters(selection_matrix)
        self.update_controller_parameters(parameters)

    def update_pd_gains(self, p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
        parameters = convert_pd_gains_to_parameters(p_gains, d_gains)
        self.update_controller_parameters(parameters)

    def set_control_mode(self, mode="parallel"):
        parameters = {"stiffness": {}}
        if mode == "parallel":
            parameters["stiffness"].update({"use_parallel_force_position_control": True})
        elif mode == "spring-mass-damper":
            parameters["stiffness"].update({"use_parallel_force_position_control": False})
        else:
            raise ValueError("Unknown control mode %s" % mode)
        self.update_controller_parameters(parameters)

    def set_position_control_mode(self, enable=True):
        parameters = convert_selection_matrix_to_parameters(np.ones(6))
        parameters["stiffness"].update({"use_parallel_force_position_control": enable})
        self.update_controller_parameters(parameters)

    def set_hand_frame_control(self, enable):
        parameters = {"hand_frame_control": {"hand_frame_control": enable}}
        self.update_controller_parameters(parameters)

    @switch_cartesian_controllers
    def execute_compliance_control(self, trajectory: np.array, target_force: np.array, max_force_torque: list,
                                   duration: float, stop_on_target_force=False):

        # Space out the trajectory points
        trajectory = trajectory.reshape((-1, 7))  # Assuming this format [x,y,z,qx,qy,qz,qw]
        step_duration = duration / float(trajectory.shape[0])
        trajectory_index = 0

        # loop throw target trajectory
        initial_time = rospy.get_time()
        step_initial_time = rospy.get_time()

        result = DONE

        # Publish target wrench only once
        self.set_cartesian_target_wrench(target_force)

        # Publish first trajectory point
        self.set_cartesian_target_pose(trajectory[trajectory_index])

        rate = rospy.Rate(100)

        while not rospy.is_shutdown() and (rospy.get_time() - initial_time) < duration:

            current_wrench = self.get_ee_wrench(hand_frame_control=False)

            if stop_on_target_force and np.all(np.abs(current_wrench)[target_force != 0] > np.abs(target_force)[target_force != 0]):
                rospy.loginfo('Target F/T reached {}'.format(np.round(current_wrench, 3)) + ' Stopping!')
                result = STOP_ON_TARGET_FORCE
                break

            # Safety limits: max force
            if np.any(np.abs(current_wrench) > max_force_torque):
                rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(current_wrench, 3)))
                result = FORCE_TORQUE_EXCEEDED
                break

            if (rospy.get_time() - step_initial_time) > step_duration:
                step_initial_time = rospy.get_time()
                trajectory_index += 1
                if trajectory_index >= trajectory.shape[0]:
                    break
                # push next point to the controller
                self.set_cartesian_target_pose(trajectory[trajectory_index])

            rate.sleep()
        # Stop moving
        # set position control only, then fix the pose to the current one
        self.set_position_control_mode()
        self.set_cartesian_target_pose(self.end_effector())

        return result
