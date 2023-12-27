import copy
from ur_control import transformations
from geometry_msgs.msg import (
    Pose,
    Point,
    Quaternion
)

class Model(object):
    """ Model object """
    def __init__(self, name, position, orientation=[0,0,0,1], file_type='urdf', string_model=None, reference_frame="world", model_id=None):
        """
        Model representation for Gazebo spawner
        name string: name of the model as it is called in the sdf/urdf
        position array[3]: x, y, z position
        orienation array[4]: ax, ay, az, w
        file_type string: type of model sdf, urdf, or string
        string_model string: full xml representing a sdf model
        reference_frame string: frame of reference for the position/orientation of the model 
        """
        self.name = name
        self.posearr = position
        self.set_pose(position, orientation)
        self.file_type = file_type
        self.string_model = string_model
        self.reference_frame = reference_frame
        self.model_id = model_id

    def set_pose(self, position, orientation=[0,0,0,1]):
        self.posearr = position
        if len(orientation) == 3:
            self.orietarr = transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        else:
            self.orietarr = orientation
        self.orientation = Quaternion(x=self.orietarr[0], y=self.orietarr[1], z=self.orietarr[2], w=self.orietarr[3]) if isinstance(orientation, list) else Quaternion()
        self.pose = Pose(position=Point(x=position[0], y=position[1], z=position[2]), orientation=self.orientation)

    def get_rotation(self):
        return self.orietarr

    def get_pose(self):
        return self.posearr
