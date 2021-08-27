import copy
from ur_control import conversions, transformations
from geometry_msgs.msg import (
    Pose,
    Point,
    Quaternion
)

class Model(object):
    """ Model object """
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
