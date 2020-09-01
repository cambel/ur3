import copy
import rospy
import rospkg

from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)


def delete_gazebo_models(models):
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        for m in models:
            delete_model(m)

    except rospy.ServiceException as e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


class GazeboModels:
    """ Class to handle ROS-Gazebo model respawn """

    def __init__(self, model_pkg):
        self.loaded_models = []
        self._pub_model_state = rospy.Publisher('/gazebo/set_model_state',
                                                ModelState, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_callback)
        rospy.sleep(0.5)
        delete_gazebo_models(self.loaded_models)
        rospy.sleep(1.0)
        # Get Models' Path
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        packpath = rospack.get_path(model_pkg)
        self.model_path = packpath + '/models/'

    def load_models(self, models):
        for m in models:
            if m.file_type == 'urdf':
                self.load_urdf_model(m)
            elif m.file_type == 'sdf' or m.file_type == 'string':
                self.load_sdf_model(m)

    def _gazebo_callback(self, data):
        self.loaded_models = []
        for obj_name in data.name:
            if obj_name.endswith("_tmp"):
                self.loaded_models.append(obj_name)

    def reset_models(self, models):
        for m in models:
            self.reset_model(m)

    def reset_model(self, model):
        """ Delete/create model if already exists, create otherwise """
        m_id = model.model_id if model.model_id is not None else model.name
        m_id += '_tmp'
        if m_id in self.loaded_models:
            delete_gazebo_models([m_id])
            rospy.sleep(0.5)
        self.load_models([model])

    def update_models_state(self, models):
        for m in models:
            self.update_model_state(m)

    def update_model_state(self, model):
        m_id = model.model_id if model.model_id is not None else model.name
        m_id += '_tmp'
        if m_id in self.loaded_models:
            model_state = ModelState(model_name=m_id, pose=model.pose, reference_frame=model.reference_frame)
            for _ in range(100):
                self._pub_model_state.publish(model_state)
        else:
            self.load_models([model])

    def load_urdf_model(self, model):
        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            m_id = model.model_id if model.model_id is not None else model.name
            spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            spawn_urdf(m_id+"_tmp", self.load_xml(model.name, filetype="urdf"), "/",
                       model.pose, model.reference_frame)
        except IOError:
            self.load_sdf_model(model)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def load_sdf_model(self, model):
        # Spawn model SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            m_id = model.model_id if model.model_id is not None else model.name
            if model.string_model is None:
                spawn_sdf(m_id+"_tmp", self.load_xml(model.name), "/",
                          model.pose, model.reference_frame)
            else:
                spawn_sdf(m_id+"_tmp", model.string_model, "/",
                          model.pose, model.reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

    def load_xml(self, model_name, filetype="sdf"):
        # Load File
        with open(self.model_path + model_name + "/model.%s" % filetype, "r") as table_file:
            return table_file.read().replace('\n', '')
