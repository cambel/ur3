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

import argparse
import copy
import sys
import timeit

import torch
import rospy
import numpy as np
import tf
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyResponse

from disect.cutting import Parameter, load_settings, CuttingSim, ROSVisualizer

from ur_control import conversions, spalg, transformations


class DisectSim():

    def __init__(self) -> None:

        self.wrench_pub = rospy.Publisher('/disect/wrench', WrenchStamped, queue_size=10)
        rospy.Service("/disect/step_simulation", Empty, self.step_simulation)
        rospy.Service("/disect/load", Empty, self.load)
        rospy.Service("/disect/reset", Empty, self.reset)
        rospy.Subscriber('/disect/knife/odom', Odometry, self.update_knife_pose, queue_size=10)

        self.parameters = {
            "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 8000),
            "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
            "sdf_ke": Parameter("sdf_ke", 500, 200., 8000, individual=True),
            "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
            "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
            "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
        }

        self.root_directory = "/root/o2ac-ur/disect/"

        self.optimized_params = [
            "log/optuna_potato_param_inference_dt2e-05_20230905-2345",
            "log/optuna_tomato_param_inference_dt3e-05_20230905-2052",
            "log/optuna_cucumber_param_inference_dt2e-05_20230905-1824",
        ]

        self.ros_frequency = 500

        self.tf_listener = tf.TransformListener()
        rospy.sleep(1)

        msg = conversions.to_pose_stamped('cutting_board_disect', [0, 0, 0, 0, 0, 0, 1])
        disect_to_robot_base = self.tf_listener.transformPose('b_bot_base_link', msg)
        self.transform_to_robot_base = conversions.from_pose_to_list(disect_to_robot_base.pose)

    def load(self, req=None):
        opt_folder = np.random.choice(self.optimized_params)
        settings = load_settings(f"{self.root_directory}/{opt_folder}/settings.json")
        settings.sim_dt = 4e-5
        settings.sim_substeps = 500
        settings.initial_y = 0.1  # center of knife + actual desired height
        self.sim = CuttingSim(settings, experiment_name="dual_sim", parameters=self.parameters, adapter='cuda',
                              requires_grad=False, root_directory=self.root_directory)
        self.sim.cut()

        # Load optimized/pretrained parameters
        pretrained_params = f"{self.root_directory}/{opt_folder}/best_optuna_optimized_tensors.pkl"
        self.sim.load_optimized_parameters(pretrained_params, verbose=True)

        self.sim.init_parameters()
        self.sim.state = self.sim.model.state()
        self.sim.assign_parameters()
        self.start_model = copy.copy(self.sim.model)
        return EmptyResponse()

    def reset(self, req=None):
        self.sim.model = copy.copy(self.start_model)
        self.sim.state = self.sim.model.state()
        self.sim.sim_time = 0.0
        self.sim.motion.reset()
        return EmptyResponse()

    def step_simulation(self, req=None):
        substeps = int((1./self.ros_frequency) / self.sim.sim_dt)
        # rospy.loginfo(f"substeps: {substeps}")
        # start_time = timeit.default_timer()

        for _ in range(substeps):
            self.sim.simulation_step()

        if hasattr(self.sim.state, 'knife_f'):
            knife_f = torch.sum(self.sim.state.knife_f, dim=0).detach().cpu().numpy()
            knife_ft = np.concatenate((knife_f, np.zeros(3)))
            gz_knife_ft = spalg.convert_wrench(knife_ft, self.transform_to_robot_base)
            self.publish_tf(gz_knife_ft)

        # rospy.loginfo(f"total computation time: {timeit.default_timer() - start_time}")
        return EmptyResponse()

    def publish_wrench(self, ft):
        print("FT", ft)
        msg = WrenchStamped()
        msg.wrench = conversions.to_wrench(ft)
        self.wrench_pub.publish(msg)

    def update_knife_pose(self, msg):
        knife_pose = conversions.from_pose_to_list(msg.pose.pose)

        knife_lin_vel = conversions.from_vector3(msg.twist.twist.linear)
        knife_ang_vel = conversions.from_vector3(msg.twist.twist.angular)

        # Compensate for position of blade
        knife_pose[1] += self.sim.knife.spine_height/2.

        # Ignore translation in x
        knife_pose[0] = 0.0
        knife_lin_vel[0] = 0.0
        # Ignore rotation in y and z
        rot_euler = np.array(transformations.euler_from_quaternion(knife_pose[3:]))
        knife_ang_vel[1] = 0.0
        knife_ang_vel[2] = 0.0
        rot_euler[1] = 0.0
        rot_euler[2] = 0.0
        knife_rot = transformations.quaternion_from_euler(*rot_euler)

        self.sim.motion.set_position(knife_pose[:3])
        self.sim.motion.set_rotation(knife_rot)
        self.sim.motion.set_linear_velocity(knife_lin_vel)
        self.sim.motion.set_angular_velocity(knife_ang_vel)


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument('-v', '--visualizer', action='store_true', help='Start disect with visualizer')
    args = parser.parse_args()

    if not args.visualizer:
        rospy.init_node('disect_simulation')
        s = DisectSim()
        # s.load()
        # s.step_simulation()
        rospy.spin()
    else:
        from PyQt5 import Qt
        rospy.init_node("disect_visualizer")
        app = Qt.QApplication(sys.argv)
        window = ROSVisualizer()
        sys.exit(app.exec_())

main()
