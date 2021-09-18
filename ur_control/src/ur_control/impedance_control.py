# Copyright (c) 2018-2021, Cristian Beltran.  All rights reserved.
#
# Cristian Beltran and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from Cristian Beltran is strictly prohibited.
#
# Author: Cristian Beltran

import numpy as np
import rospy


class AdmittanceModel():
    """
        Admittance controller
        3 different implementation are available
    """
    def __repr__(self):
        return "AdmittanceModel()"
    def __str__(self):
        return "Method: %s m: %s k: %s b: %s dt: %s" % (self.method, self.m, self.k, self.b, self.dt)

    def __init__(self, inertia, stiffness, damper, dt, method="discretization"):
        """init controller"""

        self.method = method
        self.set_constants(inertia, stiffness, damper, dt, reset=True)

    def reset(self):
        if self.method == "traditional":
            # [dx(t-1), dx(t-2)]
            self.dx_hist = [0, 0]

        elif self.method == "discretization":
            # [x(t-1), x(t-2)]
            self.dx_hist = [0.0, 0.0]
            # [f(t-1), f(t-2)]
            self.fc_hist = [0.0, 0.0]

        elif self.method == "integration":
            # position [(t-1)]
            self.position = 0.0
            # velocity [(t-1)]
            self.velocity = 0.0
            # acceleration [(t-1)]
            self.accelaration = 0.0
            # force [(t-1)]
            self.force = 0.0
        else:
            raise AttributeError("Method not supported")

    def set_constants(self, inertia, stiffness, damper, dt, reset=False):
        """ set impedance parameters """
        self.m = inertia
        self.k = stiffness
        self.b = damper
        self.T = self.dt = dt

        # compute once only
        if self.method == "traditional":
            self.T_2 = self.T**2
            self.b_t = self.b*self.T
            self.denominator = self.m + self.b_t + self.k*self.T_2
        if self.method == "discretization":
            self.T_2 = self.T**2
            self.km_term = (2*self.k*self.T_2 - 8*self.m)
            self.mbk_term1 = (4*self.m - 2*self.b*self.T + self.k*self.T_2)
            self.mbk_term2 = (4*self.m + 2*self.b*self.T + self.k*self.T_2)
        elif self.method == "integration":
            self.m_inv = np.reciprocal(self.m)

        if reset:
            self.reset()

    def control(self, fc):
        """ compute impedance step """
        if self.method == "traditional":
            return self.traditional_control(fc)
        elif self.method == "discretization":
            return self.discretization_control(fc)
        elif self.method == "integration":
            return self.integration_control(fc)

    def traditional_control(self, fc):
        """
        Implementation based on:
        A Tutorial Survey and Comparison of Impedance Control on Robotic Manipulation

        dx = (f*T^2 + B*T*dx(k-1) + M(2dx(k-1) - dx(k-2))) / (M+B*T+K*T^2)
        """
        deltax = (fc*self.T_2 + self.b_t*self.dx_hist[0] + self.m*(2*self.dx_hist[0]-self.dx_hist[1])) / self.denominator
        self.dx_hist = [deltax, self.dx_hist[0]]
        return deltax

    def discretization_control(self, fc):
        """
        Discretization method using Tustin's approximation
        Sharon, N. Hogan, and D. E. Hardt, The macro/micro manipulator:
        An improved architecture for robot control, Robot. Comput. Integr.
        Manuf., vol. 10, no. 3, pp. 209222, Jun. 1993
        x(k) = [Ts^2*f(k)+2Ts^2*f(k−1)+Ts^2f(k−2)−(2KTs^2 −
                8M)x(k−1)−(4M−2BTs+KTs^2)x(k−2))]/(4M+2BTs+KTs^2)
        """
        x = self.dx_hist
        f = self.fc_hist

        deltax = (self.T_2*fc + 2*self.T_2*f[0] + self.T_2*f[1] -
                  self.km_term*x[0] - (self.mbk_term1)*x[1]) / self.mbk_term2
        self.dx_hist = [deltax, x[0]]
        self.fc_hist = [fc, f[0]]
        return deltax

    def integration_control(self, fc):
        """
        F. Caccavale, C. Natale, B. Siciliano, and L. Villani, Integration for
        the next generation, IEEE Robot. Autom. Mag., vol. 12, no. 3, pp.
        5364, Sep. 2005.
        """
        delta_acc = self.m_inv*(fc - self.k*self.position - self.b*self.velocity)
        delta_vel = (self.T/2)*(delta_acc + self.accelaration) + self.velocity
        delta_pos = (self.T/2)*(delta_vel + self.velocity) + self.position
        self.position = delta_pos
        self.accelaration = delta_vel
        self.accelaration = delta_acc
        return delta_pos