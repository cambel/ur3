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

from ur_control import conversions
from geometry_msgs.msg import Pose

class Model(object):
    """ Gazebo Model object """
    def __init__(self, name, pose, file_type='urdf', string_model=None, reference_frame="world", model_id=None):
        """
        Model representation for Gazebo spawner
        name string: name of the model as it is called in the sdf/urdf
        pose array[6 or 7] or Pose: object pose
        file_type string: type of model sdf, urdf, or string
        string_model string: full xml representing a sdf model
        reference_frame string: frame of reference for the position/orientation of the model 
        """
        self.name = name
        self.pose = pose if isinstance(pose, Pose) else conversions.to_pose(pose)
        self.file_type = file_type
        self.string_model = string_model
        self.reference_frame = reference_frame
        self.model_id = model_id

    def set_pose(self, pose):
        self.pose = pose if isinstance(pose, Pose) else conversions.to_pose(pose)

    def get_rotation(self):
        return conversions.from_pose_to_list(self.pose)[3:]

    def get_pose(self):
        return conversions.from_pose_to_list(self.pose)[:3]
