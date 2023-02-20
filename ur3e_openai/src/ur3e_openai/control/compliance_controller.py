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

from ur_control import spalg, transformations
from ur3e_openai.control import controller


class ComplianceController(controller.Controller):
    def __init__(self, arm, agent_control_dt,
                 robot_control_dt,
                 n_actions):
        controller.Controller.__init__(self, arm, agent_control_dt, robot_control_dt, n_actions)

    def start(self):
        self.ur3e_arm.auto_switch_controllers = False
        self.ur3e_arm.activate_cartesian_controller()

    def stop(self):
        self.ur3e_arm.auto_switch_controllers = False
        self.ur3e_arm.activate_joint_trajectory_controller()

    def reset(self):
        pass

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
        if action_type == "parallel":
            # B. selection matrix + pd_gains
            self.set_parallel_parameters(actions)
        if action_type == "slicing":
            # C. slicing
            self.set_slicing_parameters(actions[:4]) 
            # Compute target pose with an attractor or with a "speed" action
            attractor_strength = np.interp(actions[-3:], [-1, 1], [0, 1.0])
            # remaining distance scale by attractor strength
            target_pose[:3] = self.ur3e_arm.end_effector()[:3] + ((target[:3] - self.ur3e_arm.end_effector()[:3]) * attractor_strength )
            # print(np.round(target_pose[:3], 4), np.round(target_pose[:3]- self.ur3e_arm.end_effector()[:3], 4))
        if action_type == "slicing_parallel":
            # C. slicing
            self.set_slicing_parallel_parameters(actions)
            f_act = np.interp(actions[-1], [-1, 1], [-15, 15])
            target_wrench = np.array([0, 0, f_act, 0, 0, 0])

        ### The action decides where to move next ###
        # motion_acts = actions[-6:]
        # motion_action = np.array([np.interp(motion_acts[i], [-1, 1], [-1*self.max_twist[i], self.max_twist[i]]) for i in range(6)])
        # print(np.round(motion_action, 4))
        # current_pose = self.ur3e_arm.end_effector()
        # target_pose = transformations.transform_pose(current_pose, actions[-6:], rotated_frame=False)

        ## go to a defined target pose ###
        return self.ur3e_arm.execute_compliance_control(trajectory=target_pose,
                                                        target_wrench=target_wrench,
                                                        max_force_torque=self.max_force_torque,
                                                        duration=self.agent_control_dt,
                                                        auto_stop=False,
                                                        scale_up_error=True,
                                                        max_scale_error=0.05)

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

    def set_slicing_parallel_parameters(self, actions):
        selection_matrix = np.array([1, 1, 0.0, 1.0, 0.5, 1])

        p_gains_z = np.interp(actions[0], [0, 1], [0.005, 0.05])
        p_gains_ay = np.interp(actions[1], [0, 1], [0.1, 1.5])
        p_gains_act = np.array([0.01, 0.01, p_gains_z, 1.5, p_gains_ay, 1.5])

        self.ur3e_arm.update_selection_matrix(selection_matrix)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("parallel")

    def set_slicing_parameters(self, actions):
        """ 
            Approach:
            force/torque control of z-axis and rotation about x-axis
            position control for remaining directions

            Target position will be given.

            Goal: optimize force control for speed and minimizing contact force
        """
        # w.r.t end effector link
        stiff_x = np.interp(actions[0], [0, 1], [100, 4000])
        stiff_ay = np.interp(actions[1], [0, 1], [5, 100])
        stiff_act = np.array([stiff_x, 2000, 2000, 40, stiff_ay, 40], dtype=int)
        # stiff_act = np.array([500, 500, 500, 20, 20, 20 ])

        # w.r.t base link
        p_gains_z = np.interp(actions[2], [0, 1], [0.005, 0.05])
        p_gains_ay = np.interp(actions[3], [0, 1], [0.1, 1.5])
        p_gains_act = np.array([0.01, 0.01, p_gains_z, 1.5, p_gains_ay, 1.5])
        # p_gains_act = np.array([0.01, 0.01, 0.01, 1.5, 1.5, 1.5])

        self.ur3e_arm.update_stiffness(stiff_act)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("spring-mass-damper")

    def compute_target_wrench(self):
        # transform = self.ur3e_arm.kdl.get_transform_between_links("b_bot_wrist_3_link", self.ur3e_arm.ee_link)
        # return spalg.convert_wrench(self.desired_force_torque, transform)
        return self.desired_force_torque
