#!/usr/bin/env python

import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, \
                                        LoadController, UnloadController, ListControllers

class ControllersConnection():
    def __init__(self, namespace=None):
        self.controllers_list = []

        if namespace:
            self.switch_service_name = '/' + namespace + '/controller_manager/switch_controller'
        else:
            self.switch_service_name = '/controller_manager/switch_controller'
        self.switch_service = rospy.ServiceProxy(self.switch_service_name, SwitchController)
        self.load_service = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        self.unload_service = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        self.list_controllers_service = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        
    def get_loaded_controllers(self):
        self.list_controllers_service.wait_for_service(0.1)
        self.controllers_list = []
        try:
            result = self.list_controllers_service()
            for controller in result.controller:
                self.controllers_list.append(controller.name)
        except:
            pass

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

    def switch_controllers(self, controllers_on, controllers_off,
                           strictness=1):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        try:
            self.switch_service.wait_for_service(0.1)
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = controllers_on
            switch_request_object.stop_controllers = controllers_off
            switch_request_object.strictness = strictness

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

            return switch_result.ok

        except Exception as e:
            rospy.logerr("Switch controllers service call failed: %s" % e)

            return None

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[], controllers_off=self.controllers_list)

        rospy.logdebug("Deactivated Controlers")

        if result_off_ok:
            rospy.logdebug("Activating Controlers")
            result_on_ok = self.switch_controllers(
                controllers_on=self.controllers_list, controllers_off=[])
            if result_on_ok:
                rospy.logdebug("Controllers Reseted==>" +
                               str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok==>" + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok==>" + str(result_off_ok))

        return reset_result

    def update_controllers_list(self, new_controllers_list):

        self.controllers_list = new_controllers_list