#!/usr/bin/env python
import gym
from .task_envs.task_envs_list import register_environment
import roslaunch
import rospy
import rospkg
import os

def load_environment(task_and_robot_environment_name,
                                timestep_limit_per_episode=10000):
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
    rospy.logwarn(
        "Env: {} will be imported".format(task_and_robot_environment_name))
    result = register_environment(
        task_env=task_and_robot_environment_name,
        timestep_limit_per_episode=timestep_limit_per_episode)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..." +
                      str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
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