#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, StepControlRequest, StepControlResponse, StepControl
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3


class ConnectionBase():
    def pause(self):
        raise NotImplementedError()

    def unpause(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class RobotConnection(ConnectionBase):
    def __init__(self):
        pass

    def pause(self):
        pass

    def unpause(self):
        pass

    def reset(self):
        pass


class GazeboConnection(ConnectionBase):
    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):
        self.step_simulation_serv = rospy.ServiceProxy('/gazebo/step_control', StepControl)
        self.step_simulation_serv.wait_for_service(1.0)

        self.reset_simulation_serv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_serv = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))

        self.set_physics = rospy.ServiceProxy(service_name,
                                              SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()
        # We always pause the simulation
        self.step_simulation_serv(StepControlRequest(steps=0))

    def pause(self):
        print("pause")
        self.step_simulation_serv(StepControlRequest(steps=1))

    def unpause(self):
        self.step_simulation_serv(StepControlRequest(steps=0))

    def reset(self):
        """
        This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
        """
        if self.reset_world_or_sim == "SIMULATION":
            rospy.logdebug("SIMULATION RESET")
            self._reset_simulation()
        elif self.reset_world_or_sim == "WORLD":
            rospy.logdebug("WORLD RESET")
            self._reset_world()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logdebug("NO RESET SIMULATION SELECTED")
        else:
            rospy.logdebug("WRONG Reset Option: {}".format(self.reset_world_or_sim))

    def _reset_simulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_serv()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

    def _reset_world(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_serv()
        except rospy.ServiceException:
            print("/gazebo/reset_world service call failed")

    def init_values(self):

        self.reset()

        if self.start_init_physics_parameters:
            rospy.logdebug("Initialising Simulation Physics Parameters")
            self._init_physics_parameters()
        else:
            rospy.logwarn("NOT Initialising Simulation Physics Parameters")

    def _init_physics_parameters(self):
        """
        We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.
        """
        self._time_step = Float64(0.001)
        self._max_update_rate = Float64(1000.0)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self._update_gravity_call()

    def _update_gravity_call(self):

        self.pause()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug("Gravity Update Result==" + str(result.success) +
                       ",message==" + str(result.status_message))

        self.unpause()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self._update_gravity_call()
