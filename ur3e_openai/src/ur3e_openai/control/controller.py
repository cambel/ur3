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

from ur3e_openai.robot_envs.utils import load_param_vars
from ur_control.fzi_cartesian_compliance_controller import CompliantController


class Controller:
    def __init__(self, arm: CompliantController,
                 agent_control_dt,
                 robot_control_dt,
                 n_actions,
                 object_centric=False):
        self.ur3e_arm = arm
        self.agent_control_dt = agent_control_dt
        self.robot_control_dt = robot_control_dt
        self.n_actions = n_actions
        self.object_centric = object_centric

        self.param_prefix = "ur3e_force_control"
        load_param_vars(self, self.param_prefix)
        self.target_force_torque = self.desired_force_torque

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        """Reset controller"""
        raise NotImplementedError()

    def act(self, action, target):
        """Execute action on controller"""
        raise NotImplementedError()
