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

from ur_control.hybrid_controller import ForcePositionController
from ur_control import utils, transformations
from ur3e_openai.control import controller
from ur3e_openai.robot_envs.utils import get_value_from_range


class ParallelController3D(controller.Controller):
    def __init__(self, arm, agent_control_dt):
        controller.Controller.__init__(self, arm, agent_control_dt)
        self.force_control_model = self._init_hybrid_controller()
        self.pd_range_type = rospy.get_param(self.param_prefix + "/pd_range_type", 'sum')

    def _init_hybrid_controller(self):
        # position PD gains
        Kp = self.base_position_kp
        Kp_pos = Kp
        Kd_pos = Kp * 0.1
        position_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

        # Force PID gains
        Kp = self.base_force_kp
        Kp_force = Kp
        Kd_force = Kp * 0.1
        Ki_force = Kp * 0.01
        force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
        return ForcePositionController(position_pd=position_pd, force_pd=force_pd, alpha=np.diag(self.alpha), dt=self.robot_control_dt)

    def reset(self):
        self.force_control_model.position_pd.reset()
        self.force_control_model.force_pd.reset()

    def act(self, action, target):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        position_action = actions[:3]
        position_action = [np.interp(position_action[i], [-1, 1], [-1*self.max_speed[i],
                                     self.max_speed[i]]) for i in range(len(position_action))]
        position_action /= self.action_scale

        target_cmd = target  # + position_action

        self.force_control_model.set_goals(position=target_cmd)
        self.set_force_signal(target_cmd)

        if self.n_actions == 6:
            actions_pds = actions[3:5]
            actions_alpha = actions[5]
        elif self.n_actions == 8:
            actions_pds = actions[3:5]
            actions_alpha = actions[5:]
        elif self.n_actions == 10:
            actions_pds = actions[3:9]
            actions_alpha = actions[9]
        elif self.n_actions == 12:
            actions_pds = actions[3:9]
            actions_alpha = actions[9:]

        self.set_pd_parameters(actions_pds)
        self.set_alpha(actions_alpha)

        # Take action
        action_result = self.ur3e_arm.set_hybrid_control(
            model=self.force_control_model,
            timeout=self.agent_control_dt,
            max_force_torque=self.max_force_torque,
            action=position_action,
        )
        return action_result

    def set_force_signal(self, target_pose):
        force_signal = self.desired_force_torque.copy()
        target_q = transformations.vector_to_pyquaternion(target_pose[3:])
        force_signal = target_q.rotate(force_signal)
        self.force_control_model.set_goals(force=force_signal)

    def set_pd_parameters(self, action):
        if len(action) == 2:
            position_kp = [get_value_from_range(action[0], self.base_position_kp[0],
                                                self.kpd_range, mtype=self.pd_range_type)]*3
            force_kp = [get_value_from_range(action[1], self.base_force_kp[0],
                                             self.kpi_range, mtype=self.pd_range_type)]*3
        else:
            assert len(action) == 6
            position_kp = [get_value_from_range(act, self.base_position_kp[i], self.kpd_range,
                                                mtype=self.pd_range_type) for i, act in enumerate(action[:3])]
            force_kp = [get_value_from_range(act, self.base_force_kp[i], self.kpi_range,
                                             mtype=self.pd_range_type) for i, act in enumerate(action[3:])]

        position_kd = np.array(position_kp) * 0.1
        force_kd = np.array(force_kp) * 0.1
        force_ki = np.array(force_kp) * 0.01

        self.force_control_model.position_pd.set_gains(Kp=position_kp, Kd=position_kd)
        self.force_control_model.force_pd.set_gains(Kp=force_kp, Kd=force_kd, Ki=force_ki)

    def set_alpha(self, action):
        if len(action) == 1:
            alpha = get_value_from_range(action, self.alpha_base, self.alpha_range)
            alpha = alpha * np.ones(3)
        else:
            assert len(action) == 3
            alpha = [get_value_from_range(act, self.alpha_base, self.alpha_range) for act in action]

        self.force_control_model.alpha = np.diag(alpha)
