#!/usr/bin/env python
import gym
from .task_envs.task_envs_list import register_environment, get_all_registered_envs
import roslaunch
import rosparam
import rospy
import rospkg
import os

def clear_gym_params(prefix):
    params = rospy.get_param_names()
    for param in params:
        if prefix in param:
            rospy.delete_param(param)

def log_ros_params(output_dir):
    params = rospy.get_param_names()
    msg = ""
    for p in params:
        if 'robot_description' in p:
            continue
        msg += "name: {}, value: {} \n".format(p, rospy.get_param(p)) 
    
    with open(output_dir + "/ros_params.log", "w") as text_file:
        text_file.write(msg)


def load_ros_params(rospackage_name, rel_path_from_package_to_file,
                           yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)
    return path_config_file


def load_environment(id, max_episode_steps=10000, register_only=False):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(id))
    result = None
    if id not in get_all_registered_envs():
        result = register_environment(
            task_env=id,
            max_episode_steps=max_episode_steps)
    else:
        result = True
    
    if register_only:
        return

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..." +
                      str(id))
        env = gym.make(id)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env


class ROSLauncher(object):
    def __init__(self,
                 rospackage_name,
                 launch_file_name,
                 ros_ws_abspath="/home/user/simulation_ws"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()

        # Check Package Exists
        pkg_path = self.rospack.get_path(rospackage_name)

        # If the package was found then we launch
        if pkg_path:
            rospy.loginfo("> Package found in workspace -->" + str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name==" +
                          str(path_launch_file_name))

            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch = roslaunch.parent.ROSLaunchParent(
                self.uuid, [path_launch_file_name])
            self.launch.start()

            rospy.loginfo("> STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS apckage ==>" + \
                str(rospackage_name)