# Copyright (c) 2018-2021, Cristian Beltran.  All rights reserved.
#
# Cristian Beltran and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from Cristian Beltran is strictly prohibited.

import rospy
from ur_control import utils, conversions, spalg
import numpy as np


class ForcePositionController(object):
    def __init__(self,
                 position_pd,
                 force_pd,
                 alpha=None,
                 dt=0.002):
        """ Force-Position controller
            alpha: [list] tradeoff between position and force signal for each direction
             """
        self.position_pd = position_pd
        self.force_pd = force_pd

        self.alpha = np.array(alpha)
        self.dt = dt

        self.error_data = list()  # data for graph
        self.update_data = list()  # data for graph

        self.safety_mode = False

        self.reset()

    def set_goals(self, position=None, force=None):
        """
           Define a goal for the position and force directions
           position: array(3)
           force: array(3)
        """
        # The position goal is the distance from
        # the current position
        if position is not None:
            assert not np.isscalar(position), "Invalid target position"
            self.target_position = position
        # TODO: The force will be constant for now
        if force is not None:
            self.target_force = force

    def reset(self):
        """ reset targets and PID params """
        self.qc = None
        self.target_position = None
        self.safety_mode = False
        self.position_pd.reset()
        self.force_pd.reset()
        self.error_data = list()
        self.update_data = list()

    def control_position(self, fc, xv):
        """ Obtains the next action from the hybrid controller
            fc: list, current wrench
            xv: list, virtual trajectory tranlsation + euler
            :return: list, angular velocity
        """
        # Force PD compensator
        Fe = -1.*self.target_force - fc  # error
        dxf_force = self.force_pd.update(error=Fe, dt=self.dt)

        # Position PD compensator
        xe = self.target_position - xv
        dxf_pos = self.position_pd.update(error=xe, dt=self.dt)

        # Sum step from force and step from position PDs
        dxf_pos = np.dot(self.alpha, dxf_pos)
        dxf_force = np.dot((np.identity(3) - self.alpha), dxf_force)
        return dxf_pos + dxf_force, dxf_pos, dxf_force

    def control_position_orientation(self, fc, xv):
        """ Obtains the next action from the hybrid controller
            fc: list, current wrench
            xv: list, virtual trajectory translation + quaternion
            :return: list, angular velocity 
        """
        # Force PD compensator
        # print("fc", fc, " ===  self.target_force", self.target_force)
        Fe = -1*self.target_force - fc  # error
        dxf_force = self.force_pd.update(error=Fe, dt=self.dt)

        # Position PD compensator
        error = spalg.translation_rotation_error(self.target_position, xv)
        dxf_pos = self.position_pd.update(error=error, dt=self.dt)

        # self.error_data.append([error, Fe])
        # self.update_data.append([dxf_pos, dxf_force])

        # Sum step from force and step from position PDs
        dxf_pos = np.dot(self.alpha, dxf_pos)
        dxf_force = np.dot((np.identity(6) - self.alpha), dxf_force)
        return dxf_pos + dxf_force, dxf_pos, dxf_force

    def control_velocity(self, fc, xv):
        """ Obtains the next action from hybrid controller 
            fc: list, current wrench
            xv: list, virtual trajectory
            :return: list, velocity (linear + angular)
        """
        # Force PD compensator
        Fe = (-1.*(self.target_force) - fc)  # error
        dxf_force = self.force_pd.update(error=Fe, dt=self.dt)

        # Position PD compensator
        xe = self.target_position - xv
        dxf_pos = self.position_pd.update(error=xe, dt=self.dt)

        # convert euler angle error to angular velocity
        T = conversions.euler_transformation_matrix(xv[3:])
        dxf_force[3:] = np.dot(T, dxf_force[3:])
        dxf_pos[3:] = np.dot(T, dxf_pos[3:])

        # Sum step from force and step from position PDs
        dxf_pos = np.dot(self.alpha, dxf_pos)
        dxf_force = np.dot((np.identity(6) - self.alpha), dxf_force)
        return dxf_pos + dxf_force, dxf_pos, dxf_force
