#!/usr/bin/env python

import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, \
    LoadController, UnloadController, ListControllers
from ur_control.utils import solve_namespace


class ControllersConnection():
    def __init__(self, namespace=None):
        self.controllers_list = []

        self.ns = solve_namespace(namespace)

        if namespace:
            prefix = '/' + namespace + 'controller_manager/'
        else:
            prefix = '/controller_manager/'

        # Only for osx sim
        if rospy.has_param("use_gazebo_sim"):
            prefix = '/controller_manager/'

        self.switch_service = rospy.ServiceProxy(prefix + 'switch_controller', SwitchController)
        self.load_service = rospy.ServiceProxy(prefix + 'load_controller', LoadController)
        self.unload_service = rospy.ServiceProxy(prefix + 'unload_controller', UnloadController)
        self.list_controllers_service = rospy.ServiceProxy(prefix + 'list_controllers', ListControllers)

    def get_loaded_controllers(self):
        self.list_controllers_service.wait_for_service(0.1)
        self.controllers_list = []
        try:
            result = self.list_controllers_service()
            for controller in result.controller:
                self.controllers_list.append(controller.name)
        except:
            pass

    def get_controller_state(self, controller_name):
        self.list_controllers_service.wait_for_service(0.1)
        try:
            result = self.list_controllers_service()
            for controller in result.controller:
                if controller.name == controller_name:
                    return controller.state
        except:
            pass
        rospy.logerr("Controller %s not found" % controller_name)
        return None

    def load_controllers(self, controllers_list):
        self.get_loaded_controllers()
        self.load_service.wait_for_service(0.1)
        for controller in controllers_list:
            if controller not in self.controllers_list:
                result = self.load_service(controller)
                rospy.loginfo('Loading controller %s. Result=%s' % (controller, result))
                if result:
                    self.controllers_list.append(controller)

    def unload_controllers(self, controllers_list):
        try:
            self.unload_service.wait_for_service(0.1)
            for controller in controllers_list:
                result = self.unload_service(controller)
                rospy.loginfo('Unloading controller %s. Result=%s' % (controller, result))
                if result:
                    self.controllers_list.remove(controller)
        except Exception as e:
            rospy.logerr("Unload controllers service call failed: %s" % e)

    def check_controllers_state(self, on_controllers, off_controllers):
        for c in on_controllers:
            if self.get_controller_state(c) != "running":
                return False
            
        for c in off_controllers:
            if self.get_controller_state(c) != "stopped":
                return False
        return True

    def switch_controllers(self, controllers_on, controllers_off,
                           strictness=1):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        if rospy.has_param("use_gazebo_sim"):
            controllers_on = [self.ns + controller for controller in controllers_on]
            controllers_off = [self.ns + controller for controller in controllers_off]

        try:
            self.switch_service.wait_for_service(0.1)
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = controllers_on
            switch_request_object.stop_controllers = controllers_off
            switch_request_object.strictness = strictness

            if self.check_controllers_state(controllers_on, controllers_off):
                return True

            switch_result = self.switch_service(switch_request_object)
            """
            [controller_manager_msgs/SwitchController]
            int32 BEST_EFFORT=1
            int32 STRICT=2
            string[] start_controllers
            string[] stop_controllers
            int32 strictness
            ---
            bool ok
            """
            rospy.logdebug("Switch Result==>" + str(switch_result.ok))

            # Check that the controllers are running before returning
            start_time = rospy.get_time()
            while (rospy.get_time() - start_time) < 5.0:
                all_running = True
                for controller in controllers_on:
                    if self.get_controller_state(controller) != "running":
                        all_running = False
                        break
                if all_running:
                    break

            return switch_result.ok

        except Exception as e:
            rospy.logerr("Switch controllers service call failed: %s" % e)

            return None

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        
        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[], controllers_off=self.controllers_list)

        rospy.logdebug("Deactivated Controllers")

        if result_off_ok:
            rospy.logdebug("Activating Controllers")
            result_on_ok = self.switch_controllers(
                controllers_on=self.controllers_list, controllers_off=[])
            if result_on_ok:
                rospy.logdebug("Controllers Reset==>" +
                               str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok==>" + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok==>" + str(result_off_ok))

        return reset_result

    def update_controllers_list(self, new_controllers_list):

        self.controllers_list = new_controllers_list
