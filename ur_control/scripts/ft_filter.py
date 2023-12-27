#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2023 Cristian Beltran
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

import argparse
import collections
import numpy as np

import rospy
from std_srvs.srv import Empty, EmptyResponse, SetBool, SetBoolResponse
from geometry_msgs.msg import WrenchStamped

from ur_control import spalg, utils, filters, conversions



class FTsensor(object):

    def __init__(self, in_topic, namespace="", out_topic=None,
                 sampling_frequency=500, cutoff=5,
                 order=2, data_window=100, timeout=3.0,
                 republish=False):

        self.ns = namespace
        self.enable_publish = republish
        self.enable_filtering = True

        self.in_topic = utils.solve_namespace(namespace + "/" + in_topic)
        if out_topic:
            self.out_topic = utils.solve_namespace(namespace + "/" + out_topic)
            self.out_tcp_topic = utils.solve_namespace(namespace + "/" + out_topic) + "tcp"
        else:
            self.out_topic = utils.solve_namespace(self.in_topic + 'filtered')
            self.out_tcp_topic = self.in_topic + "tcp"

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
        rospy.Subscriber(self.in_topic, WrenchStamped, self.raw_wrench_cb)

        # Check that the incoming topic is publishing data
        self._active = None
        if not utils.wait_for(lambda: self._active, timeout=timeout):
            rospy.logerr('Timed out waiting for {0} topic'.format(self.in_topic))
            return

        rospy.loginfo('FT filter successfully initialized')
        rospy.sleep(1)  # wait some time to fill the filter

    def add_wrench_observation(self, wrench):
        self.data_queue.append(np.array(wrench))

    def raw_wrench_cb(self, msg):
        if rospy.is_shutdown():
            return
        self._active = True
        current_wrench = conversions.from_wrench(msg.wrench)
        self.add_wrench_observation(current_wrench)
        if self.enable_publish:
            if self.enable_filtering:
                current_wrench = self.get_filtered_wrench()

            if current_wrench is not None:
                data = current_wrench - self.wrench_offset
                msg = WrenchStamped()
                msg.wrench = conversions.to_wrench(data)
                self.pub.publish(msg)

                if rospy.has_param(self.out_tcp_topic+"/pose_sensor_to_tcp"):
                    # Convert torques to force at a TCP point
                    pose_sensor_to_tcp = rospy.get_param(self.out_tcp_topic+"/pose_sensor_to_tcp")
                    tcp_wrench = data.copy()
                    tcp_wrench[:3] += spalg.sensor_torque_to_tcp_force(tcp_position=pose_sensor_to_tcp, sensor_torques=current_wrench[3:])
                    tcp_wrench[3:] = np.zeros(3)
                    msg = WrenchStamped()
                    msg.wrench = conversions.to_wrench(tcp_wrench)
                    self.pub_tcp.publish(msg)

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
    parser.add_argument('-ns', '--namespace', type=str, help='Namespace', required=False, default="")
    parser.add_argument('-t', '--ft_topic', type=str, help='FT sensor data topic', required=True)
    parser.add_argument('-ot', '--out_topic', type=str, help='Topic where filtered data will be published')
    parser.add_argument('-z', '--zero', action='store_true', help='Zero FT signal')

    args, _ = parser.parse_known_args()

    rospy.init_node('ft_filter')

    out_topic = None if not args.out_topic else args.out_topic

    ft_sensor = FTsensor(namespace=args.namespace, in_topic=args.ft_topic, out_topic=out_topic, republish=True)
    if args.zero:
        ft_sensor.update_wrench_offset()

    rospy.spin()


main()
