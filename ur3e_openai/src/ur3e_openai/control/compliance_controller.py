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

from ur_control import transformations
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

        # Options
        if action_type == "admittance":
            # A. stiffness + pd_gains
            self.set_admittance_parameters(actions)
        if action_type == "parallel":
            # B. selection matrix + pd_gains
            self.set_parallel_parameters(actions)

        return self.ur3e_arm.execute_compliance_control(trajectory=np.array(target),
                                                        target_wrench=self.compute_target_wrench(target),
                                                        max_force_torque=self.max_force_torque,
                                                        duration=self.agent_control_dt,
                                                        auto_stop=False)

    def set_parallel_parameters(self, actions):
        selection_matrix = np.interp(actions[:6], [0, 1], [0, 1])

        p_gains_trans = np.interp(actions[6:9], [0, 1], [0.005, 0.05])
        p_gains_rot = np.interp(actions[9:12], [0, 1], [0.1, 1.0])
        p_gains_act = np.concatenate([p_gains_trans, p_gains_rot])

        self.ur3e_arm.update_selection_matrix(selection_matrix)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("parallel")


    def set_admittance_parameters(self, actions):
        stiff_trans = np.interp(actions[:3], [0, 1], [100, 1000])
        stiff_rot = np.interp(actions[3:6], [0, 1], [5, 50])
        stiff_act = np.concatenate([stiff_trans, stiff_rot])

        p_gains_trans = np.interp(actions[6:9], [0, 1], [0.005, 0.05])
        p_gains_rot = np.interp(actions[9:12], [0, 1], [0.1, 1.0])
        p_gains_act = np.concatenate([p_gains_trans, p_gains_rot])

        self.ur3e_arm.update_stiffness(stiff_act)
        self.ur3e_arm.update_pd_gains(p_gains_act)
        self.ur3e_arm.set_control_mode("spring-mass-damper")

    def compute_target_wrench(self, target_pose):
        force_signal = self.desired_force_torque.copy()
        target_q = transformations.vector_to_pyquaternion(target_pose[3:])
        force_signal[:3] = target_q.rotate(force_signal[:3])
        return force_signal
