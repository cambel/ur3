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

import rospy
import rospkg

from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)

import numpy as np

from ur_control import conversions


class GazeboModels:
    """ Class to handle ROS-Gazebo model respawn """

    def __init__(self, model_pkg):
        self.loaded_models = []
        self.models_state = {}
        self._pub_model_state = rospy.Publisher('/gazebo/set_model_state',
                                                ModelState, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_callback)
        self.spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        rospy.wait_for_service('/gazebo/delete_model')

        rospy.sleep(0.5)
        self.delete_models(self.loaded_models)
        rospy.sleep(1.0)
        # Get Models' Path
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        packpath = rospack.get_path(model_pkg)
        self.model_path = packpath + '/models/'

    def delete_models(self, models, timeout=1.0):
        try:
            for m in models:
                start_time = rospy.get_time()
                while start_time - rospy.get_time() < timeout:
                    if m in self.loaded_models:
                        self.delete_model_srv(m)
                        rospy.sleep(0.15)
                    else:
                        break
        except rospy.ServiceException as e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))

    def load_models(self, models):
        for m in models:
            for _ in range(20):
                if m.file_type == 'urdf':
                    self.load_urdf_model(m)
                elif m.file_type == 'sdf' or m.file_type == 'string':
                    self.load_sdf_model(m)
                rospy.sleep(0.1)
                m_id = m.model_id if m.model_id is not None else m.name
                if m_id + '_tmp' in self.loaded_models:
                    break

    def _gazebo_callback(self, data):
        existing_models = []
        for i in range(len(data.name)):
            # get model state of all objects
            self.models_state.update({data.name[i]: conversions.from_pose_to_list(data.pose[i])})
            # store spawend objects only
            if data.name[i].endswith("_tmp"):
                existing_models.append(data.name[i])
        self.loaded_models = existing_models

    def reset_models(self, models):
        for m in models:
            self.reset_model(m)

    def reset_model(self, model):
        """ Delete/create model if already exists, create otherwise """
        m_id = model.model_id if model.model_id is not None else model.name
        m_id += '_tmp'
        if m_id in self.loaded_models:
            self.delete_models([m_id])
        rospy.sleep(1.0)
        self.load_models([model])

    def update_models_state(self, models):
        for m in models:
            self.update_model_state(m)

    def update_model_state(self, model):
        m_id = model.model_id if model.model_id is not None else model.name
        m_id += '_tmp'
        if m_id in self.loaded_models:
            model_state = ModelState(model_name=m_id, pose=model.pose, reference_frame=model.reference_frame)
            previous_pose = self.models_state.get(m_id, np.zeros(7))[:3]
            for i in range(20):
                for _ in range(10):
                    self._pub_model_state.publish(model_state)
                    rospy.sleep(0.01)
                current_pose = self.models_state.get(m_id, np.zeros(7))[:3]
                if not np.allclose(previous_pose, current_pose, rtol=0.001):
                    # print("pose updated", i)
                    break
        else:
            self.reset_model(model)

    def load_urdf_model(self, model):
        # Spawn Block URDF
        try:
            m_id = model.model_id if model.model_id is not None else model.name
            self.spawn_urdf(m_id+"_tmp", self.load_xml(model.name, filetype="urdf"), "/",
                            model.pose, model.reference_frame)
        except IOError:
            self.load_sdf_model(model)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def load_sdf_model(self, model):
        # Spawn model SDF
        try:
            m_id = model.model_id if model.model_id is not None else model.name
            if model.string_model is None:
                self.spawn_sdf(m_id+"_tmp", self.load_xml(model.name), "/",
                               model.pose, model.reference_frame)
            else:
                self.spawn_sdf(m_id+"_tmp", model.string_model, "/",
                               model.pose, model.reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

    def load_xml(self, model_name, filetype="sdf"):
        # Load File
        with open(self.model_path + model_name + "/model.%s" % filetype, "r") as table_file:
            return table_file.read().replace('\n', '')
