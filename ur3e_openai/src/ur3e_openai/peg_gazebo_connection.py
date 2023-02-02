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

#!/usr/bin/env python

from ur3e_openai.controllers_connection import ControllersConnection
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ODEPhysics
from std_srvs.srv import Empty
import rospy
import roslaunch
import rospkg
rospack = rospkg.RosPack()

PEG_SHAPES = ["cylinder", "hexagon", "cuboid", "triangular", "trapezoid", "star"]

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
    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION", controllers_list=[], controllers_on=[]):

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

        self.launch = None
        self.controllers_list = controllers_list
        self.controllers_on = controllers_on
        self.controllers = ControllersConnection()

        self.unpause_serv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_serv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_serv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_serv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # Setup the Gravity Controle system
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.set_physics.wait_for_service(1)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()

        # We always pause the simulation, important for legged robots learning
        self.pause()

    def pause(self):
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_serv()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        rospy.logdebug("PAUSING FINISH")

    def unpause(self):
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_serv()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        rospy.logdebug("UNPAUSING FiNISH")

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
        elif self.reset_world_or_sim == "ROBOT":
            rospy.logdebug("ROBOT RESET")
            self._reset_robot()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logdebug("NO RESET SIMULATION SELECTED")
        else:
            rospy.logdebug("Invalid Reset Option: {}".format(self.reset_world_or_sim))

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

    def _reset_robot(self):
        if self.need_reset_robot:
            self.unload_robot()
            self.load_robot(self.peg_shape)

    def init_values(self):
        self.peg_shape = 'cylinder'
        self.need_reset_robot = True

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

    def spawn_robot_model(self, peg_shape='cuboid'):
        """ Spawn a version of the robot with a specific gripper shaped peg """
        if peg_shape not in PEG_SHAPES:
            raise ValueError('Invalid peg shape: %s' % peg_shape)
        
        self.launch = roslaunch.scriptapi.ROSLaunch()
        launch_filepath = rospack.get_path("ur3_gazebo") + "/launch/ur_peg_alone.launch"
        # launch_filepath = rospack.get_path("ur3e_dual_gazebo") + "/launch/single_ur3e_peg_alone.launch"
        
        cli_args = [launch_filepath, 'peg_shape:=%s' % peg_shape]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.launch.parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)
        self.launch.start()

    def load_robot(self, peg_shape='cuboid'):
        """  Spawn robot and its ROS controllers """
        rospy.loginfo('Loading robot model')
        self.pause()
        self.spawn_robot_model(peg_shape)
        self.unpause()
        rospy.sleep(1)

        rospy.loginfo('Loading robot controllers')
        self.controllers.load_controllers(self.controllers_list)
        self.controllers.switch_controllers(controllers_on=self.controllers_on, controllers_off=[])
        rospy.sleep(1) # give some time to load controllers
        
        rospy.loginfo('Finish loading')

    def unload_robot(self, robot_name='robot'):
        """ Unload ROS controllers and remove model from simulation """
        self.unpause()

        rospy.loginfo("Unloading robot's controllers")
        self.controllers.switch_controllers(controllers_off=self.controllers_on, controllers_on=[])
        self.controllers.unload_controllers(self.controllers_list)
        
        rospy.loginfo("Unloading robot model")
        if self.launch:
            self.launch.stop()
        self.delete_model_srv(robot_name)
        
