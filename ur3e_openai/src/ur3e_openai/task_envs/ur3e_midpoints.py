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

import numpy as np
from ur_control import spalg, transformations
from ur3e_openai.task_envs.ur3e_task_space import UR3eTaskSpaceEnv


class UR3eMidpointsEnv(UR3eTaskSpaceEnv):
    """ Following midway points environment """

    def __init__(self):
        UR3eTaskSpaceEnv.__init__(self)
        self.current_midpoint = 0

    def _set_init_pose(self):
        self._log()
        self._add_uncertainty_error()

        self.exec_reset()

        self.ur3e_arm.set_wrench_offset(True)
        self.controller.reset()
        self.max_distance = spalg.translation_rotation_error(self.ur3e_arm.end_effector(), self.target_pos) * 1000.
        self.max_dist = None
        self.current_midpoint = 0

    def exec_reset(self):
        y_height = self.ur3e_arm.end_effector()[1]

        if y_height > self.reset_upper_height:
            self.reset_sequence(self.reset_upper_sequence, timeout=self.reset_time)
        elif y_height < self.reset_lower_height:
            self.reset_sequence(self.reset_lower_sequence, timeout=self.reset_time)
            self.reset_sequence(self.reset_upper_sequence, timeout=self.reset_time)
        else:
            self.reset_sequence(self.reset_inner_sequence, timeout=self.reset_time)
            self.reset_sequence(self.reset_lower_sequence, timeout=self.reset_time)
            self.reset_sequence(self.reset_upper_sequence, timeout=self.reset_time)

    def reset_sequence(self, seq, timeout):
        time = timeout / len(seq)
        for pose in seq:
            self.ur3e_arm.set_joint_positions(position=pose,
                                              wait=True,
                                              t=time)

    def set_position_signal(self, action):

        cpose = self.ur3e_arm.end_effector()

        dist_error = spalg.translation_rotation_error(self.target_midpoints[self.current_midpoint], cpose)
        dist_error *= [1000, 1000, 1000, 1000., 1000., 1000.]

        dist = np.linalg.norm(dist_error, axis=-1)

        if dist < self.midpoint_dist_threshold and self.current_midpoint < len(self.target_midpoints)-1:
            self.current_midpoint += 1
            self.controller.reset()

        target = self.target_midpoints[self.current_midpoint]
        if self.target_pose_uncertain:
            error = np.random.normal(scale=self.uncertainty_std, size=6)
            target = transformations.transform_pose(target, error)

        self.controller.set_goals(position=target)
