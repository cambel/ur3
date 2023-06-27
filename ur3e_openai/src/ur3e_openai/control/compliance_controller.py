# The MIT License (MIT)
#
# Copyright (c) 2018-2022 Cristian C Beltran-Hernandez
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
# Author: Cristian C Beltran-Hernandez

import sys
import rospy
import numpy as np

from ur3e_openai.control import controller
from ur_control import transformations


class ComplianceController(controller.Controller):
    def __init__(self, arm, agent_control_dt,
                 robot_control_dt,
                 n_actions):
        controller.Controller.__init__(self, arm, agent_control_dt, robot_control_dt, n_actions)
        self.added_motion_command = np.zeros(6)

    def start(self):
        self.ur3e_arm.activate_cartesian_controller()
        self.ur3e_arm.auto_switch_controllers = False
        self.ur3e_arm.async_mode = True

    def stop(self):
        self.ur3e_arm.async_mode = False
        self.ur3e_arm.set_position_control_mode(True)
        self.ur3e_arm.set_cartesian_target_pose(self.ur3e_arm.end_effector())
        rospy.sleep(0.1)
        self.ur3e_arm.auto_switch_controllers = False
        self.ur3e_arm.activate_joint_trajectory_controller()

    def reset(self):
        self.added_motion_command *= 0.0

    def act(self, action, target, action_type=None):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        target_pose = np.array(target)
        target_wrench = self.compute_target_wrench()

        # Options
        if action_type == "admittance":
            # A. stiffness + pd_gains
            self.set_admittance_parameters(actions)
        
        elif action_type == "parallel":
            # B. selection matrix + pd_gains
            self.set_parallel_parameters(actions)
        
        elif action_type == "slicing-1d-2act":
            # C. slicing
            self.set_slicing_1d_parameters(actions[:2])

        elif action_type == "slicing-1d-7act":
            # C. slicing
            self.set_slicing_parameters(actions[:4])
            # Compute target pose with an attractor or with a "speed" action
            attractor_strength = np.interp(actions[-3:], [-1, 1], [0, 1.0])
            # remaining distance scale by attractor strength
            target_pose[:3] = self.ur3e_arm.end_effector()[:3] + ((target[:3] - self.ur3e_arm.end_effector()[:3]) * attractor_strength)
        
        elif action_type == "slicing-3d":
            # C. slicing
            self.set_slicing_3d_parameters(actions[:4])
            twist_limit = self.max_twist*self.robot_control_dt

            # limit the action to a max delta translation/rotation (m/s or less)
            x_cmd = np.interp(actions[4], [-1, 1], [-1.0*twist_limit[0], twist_limit[0]])
            ay_cmd = np.interp(actions[5], [-1, 1], [-1.0*twist_limit[1], twist_limit[1]])
            self.added_motion_command[0] += x_cmd # translation in x
            self.added_motion_command[4] -= ay_cmd # rotation in ay

            target_pose = transformations.transform_pose(target_pose, self.added_motion_command, rotated_frame=False)
        
        else:
            raise ValueError("Invalid action_type %s" % action_type)

        ## go to a defined target pose ###
        return self.ur3e_arm.execute_compliance_control(trajectory=target_pose,
                                                        target_wrench=target_wrench,
                                                        max_force_torque=self.max_force_torque,
                                                        duration=self.agent_control_dt,
                                                        auto_stop=False,
                                                        scale_up_error=True,
                                                        max_scale_error=2.5)  # 0.05

    def set_parallel_parameters(self, actions):
        selection_matrix = np.interp(actions[:6], [-1, 1], [0, 1])

        p_gains_trans = np.interp(actions[6:9], [-1, 1], [0.005, 0.05])
        p_gains_rot = np.interp(actions[9:12], [-1, 1], [0.1, 1.0])
        p_gains_act = np.concatenate([p_gains_trans, p_gains_rot])

        self.ur3e_arm.update_selection_matrix(selection_matrix)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("parallel")

    def set_admittance_parameters(self, actions):
        stiff_trans = np.interp(actions[:3], [-1, 1], [100, 5000])
        stiff_rot = np.interp(actions[3:6], [-1, 1], [5, 100])
        stiff_act = np.concatenate([stiff_trans, stiff_rot])

        p_gains_trans = np.interp(actions[6:9], [-1, 1], [0.005, 0.05])
        p_gains_rot = np.interp(actions[9:12], [-1, 1], [0.1, 1.0])
        p_gains_act = np.concatenate([p_gains_trans, p_gains_rot])

        self.ur3e_arm.update_stiffness(stiff_act)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("spring-mass-damper")

    def set_slicing_1d_parameters(self, actions):
        """ 
            Approach:
            force/torque control of z-axis 
            position control for remaining directions

            Target position will be given.

            Goal: optimize force control for speed and minimizing contact force
        """
        # w.r.t end effector link
        stiff_x = np.interp(actions[0], [0, 1], [500, 5000])
        stiff_act = np.array([stiff_x, 1000, 1000, 40, 40, 40], dtype=int)

        # w.r.t base link
        p_gains_z = np.interp(actions[1], [0, 1], [0.005, 0.05])
        p_gains_act = np.array([0.1, 0.1, round(p_gains_z, 3), 1.0, 1.0, 1.0])

        self.ur3e_arm.update_selection_matrix(np.array([1, 1, 0.5, 1, 1, 1]))
        self.ur3e_arm.update_stiffness(stiff_act*2)
        self.ur3e_arm.update_pd_gains(p_gains_act*2)
        self.ur3e_arm.set_control_mode("parallel")

    def set_slicing_3d_parameters(self, actions):
        """ 
            Approach:
            force/torque control of z-axis and rotation about x-axis
            position control for remaining directions

            Target position all direction except x and ay
            
            Goal: optimize force control for speed and minimizing contact force
        """
        # w.r.t end effector link
        stiff_x = np.interp(actions[0], [0, 1], [500, 5000])
        stiff_ay = np.interp(actions[1], [0, 1], [20, 100])
        stiff_act = np.array([stiff_x, 1000, 1000, 40, stiff_ay, 40], dtype=int)
        
        # w.r.t base link
        p_gains_z = np.interp(actions[2], [0, 1], [0.01, 0.05])
        p_gains_ay = np.interp(actions[3], [0, 1], [0.1, 1.5])
        p_gains_act = np.array([0.1, 0.1, round(p_gains_z, 3), 1.0, round(p_gains_ay, 2), 1.0])
        
        self.ur3e_arm.update_selection_matrix(np.array([1, 1, 0.5, 1, 0.5, 1]))
        self.ur3e_arm.update_stiffness(stiff_act*2)
        self.ur3e_arm.update_pd_gains(p_gains_act*2)
        self.ur3e_arm.set_control_mode("parallel")

    def compute_target_wrench(self):
        return self.desired_force_torque
