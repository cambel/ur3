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

ALPHA = "alpha_only"  # only control ALPHA 6 DOFs
ALPHA_FORCE = "alpha_force"  # control ALPHA 6 DOFs + Force parameter 6 DOFs
MOTION_FORCE = "motion_force"  # control Motion (translation/rotation) 6 DOFs + Force parameter 6 DOFs
MOTION_GUIDED_FORCE = "motion_guided_force"  # control Motion (translation/rotation) 6 DOFs +  Force parameter 6 DOFs
# control Motion (translation/rotation) 6 DOFs + control ALPHA 6 DOFs + Force parameter 6 DOFs
MOTION_ALL = "motion_guided_force_and_alpha"
# control Motion (translation/rotation) 6 DOFs + Position PD 6 DOF + control ALPHA 6 DOFs + Force parameter 6 DOFs
MOTION_ALL_24 = "m24"
ALL = "all"  # control ALPHA and PIDs parameters 18 DOFs


class ParallelController(controller.Controller):
    def __init__(self, arm, agent_control_dt,
                 robot_control_dt,
                 n_actions,
                 object_centric=False,
                 ee_centric=False):
        controller.Controller.__init__(self, arm, agent_control_dt, robot_control_dt, n_actions, object_centric)
        self.force_control_model = self._init_hybrid_controller()
        self.ee_centric = ee_centric

    def _init_hybrid_controller(self):
        # position PD gains
        Kp = self.base_position_kp
        Kp_pos = Kp
        Kd_pos = Kp * 0.1
        Ki_pos = Kp * 0.0
        position_pd = utils.PID(Kp=Kp_pos, 
                                Kd=Kd_pos, 
                                Ki=Ki_pos, 
                                dynamic_pid=True, 
                                max_gain_multiplier=10)

        # Force PID gains
        Kp = self.base_force_kp
        Kp_force = Kp
        Kd_force = Kp * 0.0
        Ki_force = Kp * 0.01
        force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
        return ForcePositionController(position_pd=position_pd, 
                                       force_pd=force_pd, 
                                       alpha=np.diag(self.alpha), 
                                       dt=self.robot_control_dt)

    def reset(self):
        self.force_control_model.position_pd.reset()
        self.force_control_model.force_pd.reset()

    def act(self, action, target, action_type=ALL):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        self.set_force_signal(target)
        self.force_control_model.target_position = target

        mtype = "parallel"
        if action_type == ALPHA:
            actions_alpha = actions[:6]
            self.set_alpha(actions_alpha)
        elif action_type == ALPHA_FORCE:
            self.set_force_parameters(actions[:6])
            actions_alpha = actions[6:12]
            self.set_alpha(actions_alpha)
        elif action_type == MOTION_FORCE:
            motion_command = self.set_position_signal(actions[:6])
            self.set_force_parameters(actions[6:12])
        elif action_type == MOTION_GUIDED_FORCE:
            mtype = "parallel-guided"
            motion_command = self.set_position_signal(actions[:6])
            self.set_force_parameters(actions[6:12])
        elif action_type == MOTION_ALL:
            mtype = "parallel-guided-with-alpha"
            motion_command = self.set_position_signal(actions[:6])
            self.set_force_parameters(actions[6:12])
            actions_alpha = actions[12:18]
            self.set_alpha(actions_alpha)
        elif action_type == MOTION_ALL_24:
            mtype = "parallel-guided-with-alpha"
            motion_command = self.set_position_signal(actions[:6])
            self.set_position_parameters(actions[6:12])
            self.set_force_parameters(actions[12:18])
            actions_alpha = actions[18:24]
            self.set_alpha(actions_alpha)
        elif action_type == ALL:
            self.set_position_parameters(actions[:6])
            self.set_force_parameters(actions[6:12])
            actions_alpha = actions[12:18]
            self.set_alpha(actions_alpha)

        # Take action
        if action_type in (MOTION_FORCE, MOTION_GUIDED_FORCE, MOTION_ALL, MOTION_ALL_24):
            action_result = self.ur3e_arm.execute_control(motion_command=motion_command,
                                                          model=self.force_control_model,
                                                          max_force_torque=self.max_force_torque,
                                                          timeout=self.agent_control_dt,
                                                          mtype=mtype,
                                                          object_centric=self.object_centric,
                                                          ee_centric=self.ee_centric)
        else:
            action_result = self.ur3e_arm.execute_hybrid_control(
                model=self.force_control_model,
                timeout=self.agent_control_dt,
                max_force_torque=self.max_force_torque,
                object_centric=self.object_centric
            )
        return action_result

    def set_position_signal(self, actions):
        # limit the action to a max delta translation/rotation (1m/s or less)
        action = np.array([np.interp(actions[i], [-1, 1], [-1*self.max_twist[i], self.max_twist[i]])
                          for i in range(len(actions))])
        return action

    def set_force_signal(self, target_pose):
        force_signal = self.desired_force_torque.copy()
        target_q = transformations.vector_to_pyquaternion(target_pose[3:])
        force_signal[:3] = target_q.rotate(force_signal[:3])
        self.force_control_model.set_goals(force=force_signal)

    def set_position_parameters(self, action):
        position_kp = [get_value_from_range(
            act, self.base_position_kp[i], self.kpd_range, mtype=self.pd_range_type) for i, act in enumerate(action[:6])]
        position_kd = np.array(position_kp) * 0.1
        self.force_control_model.position_pd.set_gains(
            Kp=position_kp, Kd=position_kd)

    def set_force_parameters(self, action):
        force_kp = [get_value_from_range(
            act, self.base_force_kp[i], self.kpi_range, mtype=self.pd_range_type) for i, act in enumerate(action[:6])]
        force_kd = np.array(force_kp) * 0.1
        force_ki = np.array(force_kp) * 0.01
        self.force_control_model.force_pd.set_gains(
            Kp=force_kp, Kd=force_kd, Ki=force_ki)

    def set_alpha(self, action):
        alpha = np.interp(action, [-1, 1], [0, 1])
        self.force_control_model.alpha = np.diag(alpha)
