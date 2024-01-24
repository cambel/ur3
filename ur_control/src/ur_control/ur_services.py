
import rospy
import time

import controller_manager_msgs.msg
import std_srvs.srv
from ur_control import conversions
from ur_control.utils import solve_namespace
import ur_dashboard_msgs.srv
import ur_msgs.srv

from std_msgs.msg import Bool


def check_for_real_robot(func):
    '''Decorator that validates the real robot is used or no'''

    def wrap(*args, **kwargs):
        if args[0].use_real_robot:
            return func(*args, **kwargs)
        rospy.logdebug("Ignoring function %s since no real robot is being used" % func.__name__)
        return True
    return wrap


class URServices():
    """ 
    Universal Robots driver specific services
    """

    def __init__(self, namespace):

        self.use_real_robot = rospy.get_param("use_real_robot", False)

        self.ns = solve_namespace(namespace)

        self.ur_dashboard_clients = {
            "get_loaded_program":     rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/get_loaded_program', ur_dashboard_msgs.srv.GetLoadedProgram),
            "program_running":        rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/program_running', ur_dashboard_msgs.srv.IsProgramRunning),
            "load_program":           rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/load_program', ur_dashboard_msgs.srv.Load),
            "play":                   rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/play', std_srvs.srv.Trigger),
            "stop":                   rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/stop', std_srvs.srv.Trigger),
            "quit":                   rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/quit', std_srvs.srv.Trigger),
            "connect":                rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/connect', std_srvs.srv.Trigger),
            "close_popup":            rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/close_popup', std_srvs.srv.Trigger),
            "unlock_protective_stop": rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/unlock_protective_stop', std_srvs.srv.Trigger),
            "is_in_remote_control":   rospy.ServiceProxy(self.ns + 'ur_hardware_interface/dashboard/is_in_remote_control', ur_dashboard_msgs.srv.IsInRemoteControl),
        }

        self.set_payload_srv = rospy.ServiceProxy(self.ns + 'ur_hardware_interface/set_payload', ur_msgs.srv.SetPayload)
        self.speed_slider = rospy.ServiceProxy(self.ns + 'ur_hardware_interface/set_speed_slider', ur_msgs.srv.SetSpeedSliderFraction)

        self.set_io = rospy.ServiceProxy(self.ns + 'ur_hardware_interface/set_io', ur_msgs.srv.SetIO)

        self.sub_status_ = rospy.Subscriber(self.ns + 'ur_hardware_interface/robot_program_running', Bool, self.ros_control_status_callback)
        self.service_proxy_list = rospy.ServiceProxy(self.ns + 'controller_manager/list_controllers', controller_manager_msgs.srv.ListControllers)
        self.service_proxy_switch = rospy.ServiceProxy(self.ns + 'controller_manager/switch_controller', controller_manager_msgs.srv.SwitchController)

        self.sub_robot_safety_mode = rospy.Subscriber(self.ns + 'ur_hardware_interface/safety_mode', ur_dashboard_msgs.msg.SafetyMode, self.safety_mode_callback)

        self.ur_ros_control_running_on_robot = False
        self.robot_safety_mode = None
        self.robot_status = dict()

    @check_for_real_robot
    def safety_mode_callback(self, msg):
        self.robot_safety_mode = msg.mode

    @check_for_real_robot
    def ros_control_status_callback(self, msg):
        self.ur_ros_control_running_on_robot = msg.data

    @check_for_real_robot
    def is_running_normally(self):
        """
        Returns true if the robot is running (no protective stop, not turned off etc).
        """
        return self.robot_safety_mode == 1 or self.robot_safety_mode == 2  # Normal / Reduced

    @check_for_real_robot
    def is_protective_stopped(self):
        """
        Returns true if the robot is in protective stop.
        """
        return self.robot_safety_mode == 3

    @check_for_real_robot
    def unlock_protective_stop(self):
        if not self.use_real_robot:
            return True

        service_client = self.ur_dashboard_clients["unlock_protective_stop"]
        request = std_srvs.srv.TriggerRequest()
        start_time = time.time()
        rospy.loginfo("Attempting to unlock protective stop of " + self.ns)
        while not rospy.is_shutdown():
            response = service_client.call(request)
            if time.time() - start_time > 20.0:
                rospy.logerr("Timeout of 20s exceeded in unlock protective stop")
                break
            if response.success:
                break
            rospy.sleep(0.2)
        self.ur_dashboard_clients["stop"].call(std_srvs.srv.TriggerRequest())
        if not response.success:
            rospy.logwarn("Could not unlock protective stop of " + self.ns + "!")
        return response.success

    @check_for_real_robot
    def set_speed_scale(self, scale):
        try:
            self.speed_slider(ur_msgs.srv.SetSpeedSliderFractionRequest(speed_slider_fraction=scale))
        except:
            rospy.logerr("Failed to communicate with Dashboard when setting speed slider")
            return False

    @check_for_real_robot
    def set_payload(self, mass, center_of_gravity):
        """ 
            mass float
            center_of_gravity list[3] 
        """
        self.activate_ros_control_on_ur()
        try:
            payload = ur_msgs.srv.SetPayloadRequest()
            payload.payload = mass
            payload.center_of_gravity = conversions.to_vector3(center_of_gravity)
            self.set_payload_srv(payload)
            return True
        except Exception as e:
            rospy.logerr("Exception trying to set payload: %s" % e)
        return False

    @check_for_real_robot
    def wait_for_control_status_to_turn_on(self, waittime):
        start = rospy.Time.now()
        elapsed = rospy.Time.now() - start
        while not self.ur_ros_control_running_on_robot and elapsed < rospy.Duration(waittime) and not rospy.is_shutdown():
            rospy.sleep(.1)
            elapsed = rospy.Time.now() - start
            if self.ur_ros_control_running_on_robot:
                return True
        return False

    @check_for_real_robot
    def activate_ros_control_on_ur(self, recursion_depth=0):
        if not self.use_real_robot:
            return True

        # Check if URCap is already running on UR
        if self.ur_ros_control_running_on_robot:
            self.set_speed_scale(scale=1.0)  # Set speed to max always
            return True
        else:
            rospy.loginfo("Robot program not running for " + self.ns)

        try:
            response = self.ur_dashboard_clients["is_in_remote_control"].call()
            if not response.in_remote_control:
                rospy.logerr("Unable to automatically activate robot. Manually activate the robot by pressing 'play' in the polyscope or turn ON the remote control mode.")
                return False
        except:
            pass

        rospy.logwarn(f"Attempt to reconnect # {recursion_depth+1}")

        if recursion_depth > 10:
            rospy.logerr("Tried too often. Breaking out.")
            rospy.logerr("Could not start UR ROS control.")
            raise Exception("Could not activate ROS control on robot " + self.ns + ". Breaking out. Is the UR in Remote Control mode and program installed with correct name?")

        if rospy.is_shutdown():
            return False

        program_loaded = self.check_loaded_program()

        if not program_loaded:
            rospy.logwarn("Could not load.")
        else:
            # Run the program
            rospy.loginfo("Running the program (play)")
            try:
                response = self.ur_dashboard_clients["play"].call(std_srvs.srv.TriggerRequest())
            except:
                pass
            if self.wait_for_control_status_to_turn_on(2.0):
                return True
            else:
                rospy.logwarn("Failed to start program")

        rospy.logwarn("Trying to reconnect dashboard client and then activating again.")
        # Try to connect to dashboard if first try failed
        try:
            rospy.logdebug("Try to quit before connecting.")
            response = self.ur_dashboard_clients["quit"].call()
            rospy.sleep(1)
            rospy.logdebug("Try to connect to dashboard service.")
            response = self.ur_dashboard_clients["connect"].call()
            rospy.sleep(1)
            if response.success:
                rospy.logdebug("Try to stop program.")
                response = self.ur_dashboard_clients["stop"].call()
                rospy.sleep(1)
                if response.success:
                    rospy.logdebug("Try to play program.")
                    response = self.ur_dashboard_clients["play"].call()
                    rospy.sleep(1)
        except:
            rospy.logwarn("Dashboard service did not respond! (2)")
            pass

        if self.wait_for_control_status_to_turn_on(2.0):
            if self.check_for_dead_controller_and_force_start():
                rospy.loginfo("Successfully activated ROS control on robot " + self.ns)
                self.set_speed_scale(scale=1.0)  # Set speed to max always
                return True
        else:
            return self.activate_ros_control_on_ur(recursion_depth=recursion_depth+1)

    @check_for_real_robot
    def check_loaded_program(self):
        try:
            # Load program if it not loaded already
            response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
            if response.program_name == '/programs/ROS_external_control.urp':
                return True
            else:
                rospy.loginfo("Currently loaded program was:  " + response.program_name)
                rospy.loginfo("Loading ROS control on robot " + self.ns)
                request = ur_dashboard_msgs.srv.LoadRequest()
                request.filename = "ROS_external_control.urp"
                response = self.ur_dashboard_clients["load_program"].call(request)
                if response.success:  # Try reconnecting to dashboard
                    return True
                else:
                    rospy.logerr("Could not load the ROS_external_control.urp URCap. Is the UR in Remote Control mode and program installed with correct name?")
                for i in range(10):
                    rospy.sleep(0.2)
                    # rospy.loginfo("After-load check nr. " + str(i))
                    response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
                    # rospy.loginfo("Received response: " + response.program_name)
                    if response.program_name == '/programs/ROS_external_control.urp':
                        break
        except:
            rospy.logwarn("Dashboard service did not respond!")
        return False

    @check_for_real_robot
    def check_for_dead_controller_and_force_start(self):
        list_req = controller_manager_msgs.srv.ListControllersRequest()
        switch_req = controller_manager_msgs.srv.SwitchControllerRequest()
        rospy.loginfo("Checking for dead controllers for robot " + self.ns)
        list_res = self.service_proxy_list.call(list_req)
        for c in list_res.controller:
            if c.name == "scaled_pos_joint_traj_controller":
                if c.state == "stopped":
                    # Force restart
                    rospy.logwarn("Force restart of controller")
                    switch_req.start_controllers = ['scaled_pos_joint_traj_controller']
                    switch_req.strictness = 1
                    switch_res = self.service_proxy_switch.call(switch_req)
                    rospy.sleep(1)
                    return switch_res.ok
                else:
                    rospy.loginfo("Controller state is " + c.state + ", returning True.")
                    return True

    @check_for_real_robot
    def load_and_execute_program(self, program_name="", recursion_depth=0, skip_ros_activation=False):
        if not skip_ros_activation:
            self.activate_ros_control_on_ur()
        if not self.load_program(program_name, recursion_depth):
            return False
        return self.execute_loaded_program()

    @check_for_real_robot
    def load_program(self, program_name="", recursion_depth=0):
        if not self.use_real_robot:
            return True

        if recursion_depth > 10:
            rospy.logerr("Tried too often. Breaking out.")
            rospy.logerr("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
            return False

        load_success = False
        try:
            # Try to stop running program
            self.ur_dashboard_clients["stop"].call(std_srvs.srv.TriggerRequest())
            rospy.sleep(.5)

            # Load program if it not loaded already
            response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
            # print("response:")
            # print(response)
            if response.program_name == '/programs/' + program_name:
                return True
            else:
                rospy.loginfo("Loaded program is different %s. Attempting to load new program %s" % (response.program_name, program_name))
                request = ur_dashboard_msgs.srv.LoadRequest()
                request.filename = program_name
                response = self.ur_dashboard_clients["load_program"].call(request)
                if response.success:  # Try reconnecting to dashboard
                    load_success = True
                    return True
                else:
                    rospy.logerr("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
        except:
            rospy.logwarn("Dashboard service did not respond to load_program!")
        if not load_success:
            rospy.logwarn("Waiting and trying again")
            rospy.sleep(3)
            try:
                if recursion_depth > 0:  # If connect alone failed, try quit and then connect
                    response = self.ur_dashboard_clients["quit"].call()
                    rospy.logerr("Program could not be loaded on UR: " + program_name)
                    rospy.sleep(.5)
            except:
                rospy.logwarn("Dashboard service did not respond to quit! ")
                pass
            response = self.ur_dashboard_clients["connect"].call()
            rospy.sleep(.5)
            return self.load_program(program_name=program_name, recursion_depth=recursion_depth+1)

    @check_for_real_robot
    def execute_loaded_program(self):
        # Run the program
        try:
            response = self.ur_dashboard_clients["play"].call(std_srvs.srv.TriggerRequest())
            if not response.success:
                rospy.logerr("Could not start program. Is the UR in Remote Control mode and program installed with correct name?")
                return False
            else:
                rospy.loginfo("Successfully started program on robot " + self.ns)
                return True
        except Exception as e:
            rospy.logerr(str(e))
            return False

    @check_for_real_robot
    def close_ur_popup(self):
        # Close a popup on the teach pendant to continue program execution
        response = self.ur_dashboard_clients["close_popup"].call(std_srvs.srv.TriggerRequest())
        if not response.success:
            rospy.logerr("Could not close popup.")
            return False
        else:
            rospy.loginfo("Successfully closed popup on teach pendant of robot " + self.ns)
            return True
