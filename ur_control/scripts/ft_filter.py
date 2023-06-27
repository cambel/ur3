#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
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
# Author: Cristian Beltran

import copy
import argparse
import collections
import rospy

import numpy as np
from ur_control.constants import get_arm_joint_names
from ur_pykdl import ur_kinematics
from ur_control import spalg, utils, filters, conversions

from std_srvs.srv import Empty, EmptyResponse, SetBool, SetBoolResponse

from geometry_msgs.msg import WrenchStamped, Wrench
from ur_control.controllers import JointControllerBase


class FTsensor(object):

    def __init__(self, in_topic, in_topic2 = None, namespace="", out_topic=None,
                 sampling_frequency=500, cutoff=5,
                 order=2, data_window=100, timeout=3.0,
                 republish=False, gravity_compensation=False):

        self.ns = namespace
        self.enable_publish = republish
        self.enable_filtering = True
        self.gravity_compensation = gravity_compensation

        self.in_topic = utils.solve_namespace(namespace + "/" + in_topic)
        if out_topic:
            self.out_topic = utils.solve_namespace(namespace + "/" + out_topic)
            self.out_tcp_topic = utils.solve_namespace(namespace + "/" + out_topic) + "tcp"
        else:
            self.out_topic = utils.solve_namespace(self.in_topic + 'filtered')
            self.out_tcp_topic = self.in_topic + "tcp"
        self.in_topic2 = in_topic2

        if self.gravity_compensation:
            prefix = "" if not namespace else namespace + "_"
            base_link = "base_link"
            ft_sensor_link = "tool0"
            self.robot_state = JointControllerBase(namespace, timeout=1, joint_names=get_arm_joint_names(prefix))
            self.kdl = ur_kinematics(base_link=prefix + base_link, ee_link=prefix + ft_sensor_link)

            # gravity compensation
            # In base frame
            self.mass = 1.250
            self.gravity = np.array([0, 0, -0.981])
            self.center_of_mass = np.array([0.001, -0.0016, 0.05])
            self.weight_force = np.concatenate([self.mass * self.gravity, np.zeros(3)])

        rospy.loginfo("Publishing filtered FT to %s" % self.out_topic)

        # Load previous offset to zero filtered signal
        self.wrench_offset = rospy.get_param('%sft_offset' % self.out_topic, None)
        self.wrench_offset = np.zeros(6) if self.wrench_offset is None else self.wrench_offset

        # Publisher to outward topic
        self.pub = rospy.Publisher(self.out_topic, WrenchStamped, queue_size=10)
        # Publish a wrench transformed/converted to a TCP point
        self.pub_tcp = rospy.Publisher(self.out_tcp_topic, WrenchStamped, queue_size=10)

        # Service for zeroing the filtered signal
        rospy.Service(self.out_topic + "zero_ftsensor", Empty, self._srv_zeroing)
        rospy.Service(self.out_topic + "enable_publish", SetBool, self._srv_publish)
        rospy.Service(self.out_topic + "enable_filtering", SetBool, self._srv_filtering)

        # Low pass filter
        self.filter = filters.ButterLowPass(cutoff, sampling_frequency, order)

        self.data_window = data_window
        assert (self.data_window >= 5)
        self.data_queue = collections.deque(maxlen=self.data_window)

        # Subscribe to incoming topic
        self.added_wrench = np.zeros(6)
        if self.in_topic2 != None : rospy.Subscriber(self.in_topic2, WrenchStamped, self.cb_adder)
        rospy.Subscriber(self.in_topic, WrenchStamped, self.cb_raw)

        # Check that the incoming topic is publishing data
        self._active = None
        if not utils.wait_for(lambda: self._active, timeout=timeout):
            rospy.logerr('Timed out waiting for {0} topic'.format(self.in_topic))
            return

        rospy.loginfo('FT filter successfully initialized')
        rospy.sleep(1)  # wait some time to fill the filter
        # rospy.sleep(self.data_window * (1/sampling_frequency))  # wait some time to fill the filter

    def apply_gravity_compensation(self, wrench):
        compensated_wrench = wrench.copy()
        # Compute actual gravity effects in sensor frame
        gravity_component = np.zeros(6)
        gravity_component[:3] = spalg.convert_wrench(self.weight_force, self.kdl.forward(self.robot_state.get_joint_positions()))[:3]
        gravity_component[3:] = self.center_of_mass * gravity_component[:3]  # M = r x F

        rospy.loginfo_throttle(1, "gravity compensation %s" % np.round(gravity_component, 3)[:3])
        
        # Add actual gravity compensation
        compensated_wrench -= gravity_component

        return compensated_wrench

    def add_wrench_observation(self, wrench):
        self.data_queue.append(np.array(wrench))

    def cb_adder(self, msg):
        if rospy.is_shutdown():
            return
        self.added_wrench = conversions.from_wrench(msg.wrench)

    def cb_raw(self, msg):
        if rospy.is_shutdown():
            return
        self._active = True
        current_wrench = conversions.from_wrench(msg.wrench) + self.added_wrench
        self.add_wrench_observation(current_wrench)
        if self.enable_publish:
            if self.enable_filtering:
                current_wrench = self.get_filtered_wrench()

            if current_wrench is not None:
                if self.gravity_compensation:
                    data = self.apply_gravity_compensation(current_wrench)
                    data = data - self.wrench_offset
                else:
                    data = current_wrench - self.wrench_offset
                if np.any(np.isnan(data)) :
                    rospy.logerr("NaN values in the output of the filter. Ignoring.")
                    self.added_wrench = np.zeros(6)
                    self.wrench_offset = np.zeros(6)
                    self.data_queue = collections.deque(maxlen=self.data_window)
                    rospy.signal_shutdown()
                    return
                filtered_msg = WrenchStamped()
                filtered_msg.wrench = conversions.to_wrench(data)
                filtered_msg.header.frame_id = msg.header.frame_id
                self.pub.publish(filtered_msg)

                if rospy.has_param(self.out_tcp_topic+"/pose_sensor_to_tcp"):
                    # Convert torques to force at a TCP point
                    pose_sensor_to_tcp = rospy.get_param(self.out_tcp_topic+"/pose_sensor_to_tcp")
                    tcp_wrench = data.copy()
                    tcp_wrench[:3] += spalg.sensor_torque_to_tcp_force(tcp_position=pose_sensor_to_tcp, sensor_torques=current_wrench[3:])
                    tcp_wrench[3:] = np.zeros(3)
                    tcp_msg = WrenchStamped()
                    tcp_msg.wrench = conversions.to_wrench(tcp_wrench)
                    self.pub_tcp.publish(tcp_msg)

    # function to filter out high frequency signal
    def get_filtered_wrench(self):
        if len(self.data_queue) < self.data_window:
            return None
        wrench_filtered = self.filter(np.array(self.data_queue))
        return wrench_filtered[-1, :]

    def update_wrench_offset(self):
        current_wrench = self.get_filtered_wrench()
        if current_wrench is not None:
            self.wrench_offset = current_wrench
            if self.wrench_offset is not None:
                rospy.set_param('%sft_offset' % self.out_topic, self.wrench_offset.tolist())

    def set_enable_publish(self, enable):
        self.enable_publish = enable

    def set_enable_filtering(self, enable):
        self.enable_filtering = enable

    def _srv_zeroing(self, req):
        self.update_wrench_offset()
        return EmptyResponse()

    def _srv_publish(self, req):
        self.set_enable_publish(req.data)
        return SetBoolResponse(success=True)

    def _srv_filtering(self, req):
        self.set_enable_filtering(req.data)
        return SetBoolResponse(success=True)


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Filter FT signal')
    parser.add_argument('-ns', '--namespace', type=str, help='Namespace', required=False)
    parser.add_argument('-t', '--ft_topic', type=str, help='FT sensor data topic', required=True)
    parser.add_argument('-t2', '--ft_topic2', type=str, help='FT sensor data topic (2nd)', required=False, default=None)
    parser.add_argument('-ot', '--out_topic', type=str, help='Topic where filtered data will be published')
    parser.add_argument('-z', '--zero', action='store_true', help='Zero FT signal')
    parser.add_argument('-g', '--gravity_compensation', action='store_true', help='Gravity compensation applied')

    args, unknown = parser.parse_known_args()

    rospy.init_node('ft_filter')

    out_topic = None if not args.out_topic else args.out_topic

    ft_sensor = FTsensor(namespace=args.namespace, in_topic=args.ft_topic, in_topic2=args.ft_topic2, out_topic=out_topic,
                         republish=True, gravity_compensation=args.gravity_compensation)
    if args.zero:
        ft_sensor.update_wrench_offset()

    rospy.spin()


main()
