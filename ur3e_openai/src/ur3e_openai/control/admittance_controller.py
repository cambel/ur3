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
# 

import sys
import rospy
import numpy as np

from ur3e_openai.control.parallel_controller import ALL
from ur_control import utils
from ur_control.admittance_control import AdmittanceModel
from ur3e_openai.control import controller
from ur3e_openai.robot_envs.utils import get_value_from_range

ALL = "all"


class AdmittanceController(controller.Controller):
    def __init__(self, arm, agent_control_dt,
                 robot_control_dt,
                 n_actions,
                 object_centric=False):
        controller.Controller.__init__(self, arm, agent_control_dt, robot_control_dt, n_actions, object_centric)
        self.force_control_model = self._init_admittance_controller()

    def _init_admittance_controller(self):
        position_kp = self.base_position_kp
        position_kd = np.array(position_kp) * 0.1

        M = self.inertia_const * np.ones(6)
        K = self.stiffness_const * np.ones(6)
        B = np.zeros(6)
        # B = 2*self.damping_ratio*np.sqrt(M*K)

        model = AdmittanceModel(M, K, B, self.robot_control_dt, method=self.impedance_method)
        model.position_pd = utils.PID(Kp=position_kp, Kd=position_kd)
        return model

    def reset(self):
        self.force_control_model.position_pd.reset()

    def act(self, action, target, action_type=ALL):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        motion_command = self.set_position_signal(actions[:6])
        self.set_impedance_parameters(action[6:12])
        self.force_control_model.position_target = target

        # Take action
        action_result = self.ur3e_arm.execute_control(motion_command=motion_command,
                                                      model=self.force_control_model,
                                                      timeout=self.agent_control_dt,
                                                      max_force_torque=self.max_force_torque,
                                                      mtype=action_type)
        return action_result

    def set_position_signal(self, actions):
        # limit the action to a max delta translation/rotation (1m/s or less)
        action = np.array([np.interp(actions[i], [-1, 1], [-1*self.max_twist[i], self.max_twist[i]])
                          for i in range(len(actions))])
        action *= self.agent_control_dt
        return action

    def set_pd_parameters(self, action):
        if len(action) == 1:
            position_kp = [get_value_from_range(action, self.base_position_kp[0], self.kpd_range)]*6
        else:
            assert len(action) == 6
            position_kp = [get_value_from_range(act, self.base_position_kp[i], self.kpd_range,
                                                mtype=self.pd_range_type) for i, act in enumerate(action[:6])]

        position_kd = np.array(position_kp) * 0.1
        self.force_control_model.position_pd.set_gains(Kp=position_kp, Kd=position_kd)

    def set_impedance_parameters(self, action):
        K = [get_value_from_range(action[i], self.stiffness_const[i], self.param_range) for i in range(6)]
        K = np.array(K)
        M = self.inertia_const * np.ones(6)
        B = 2*self.damping_ratio*np.sqrt(M*K)
        self.force_control_model.set_constants(M, K, B, self.robot_control_dt)
