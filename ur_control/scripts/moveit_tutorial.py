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
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

import signal


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class MoveGroupPythonIntefaceTutorial(object):
    """MoveGroupPythonIntefaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonIntefaceTutorial, self).__init__()

        # BEGIN_SUB_TUTORIAL setup
        ##
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        # Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to a planning group (group of joints).  In this tutorial the group is the primary
        # arm joints in the ur3e robot, so we set the group's name to "ur3e_arm".
        # If you are using a different robot, change this value to the name of your robot
        # arm planning group.
        # This interface can be used to plan and execute motions:
        group_name = "arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        # END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = move_group.get_planning_frame()
        self.eef_link = move_group.get_end_effector_link()
        self.group_names = robot.get_group_names()

    def display_basic_info(self):
            # BEGIN_SUB_TUTORIAL basic_info
        ##
        # Getting Basic Information
        # ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        print("============ Planning frame: %s" % self.planning_frame)

        # We can also print the name of the end-effector link for this group:
        print("============ End effector link: %s" % self.eef_link)

        # We can get a list of all the groups in the robot:
        print("============ Available Planning Groups:", self.group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")
        # END_SUB_TUTORIAL

    def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        # Planning to a Joint Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^^
        # The ur3e's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
        # thing we want to do is move it to a slightly better configuration.
        # We can get the joint values from the group and adjust some of the values:
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 1.57
        joint_goal[1] = -1.57
        joint_goal[2] = 1.30
        joint_goal[3] = -1.57
        joint_goal[4] = -1.57
        joint_goal[5] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        # END_SUB_TUTORIAL

        # For testing:
        print("Current pose", move_group.get_current_pose().pose)
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        # Planning to a Pose Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^
        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        pose_goal = move_group.get_current_pose().pose
        pose_goal.position.y += -0.1

        move_group.set_pose_target(pose_goal)

        # Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

        # END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def plan_cartesian_path(self, scale=1):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        # Cartesian Paths
        # ^^^^^^^^^^^^^^^
        # You can plan a Cartesian path directly by specifying a list of waypoints
        # for the end-effector to go through. If executing  interactively in a
        # Python shell, set scale = 1.0.
        ##
        waypoints = []

        wpose = move_group.get_current_pose().pose
        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints,   # waypoints to follow
            0.01,        # eef_step
            0.0)         # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        # END_SUB_TUTORIAL

    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        # BEGIN_SUB_TUTORIAL display_trajectory
        ##
        # Displaying a Trajectory
        # ^^^^^^^^^^^^^^^^^^^^^^^
        # You can ask RViz to visualize a plan (aka trajectory) for you. But the
        # group.plan() method does this automatically so this is not that useful
        # here (it just displays the same trajectory again):
        ##
        # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # We populate the trajectory_start with our current robot state to copy over
        # any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)

        # END_SUB_TUTORIAL

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL execute_plan
        ##
        # Executing a Plan
        # ^^^^^^^^^^^^^^^^
        # Use execute if you would like the robot to follow
        # the plan that has already been computed:
        move_group.execute(plan, wait=True)

        # **Note:** The robot's current joint state must be within some tolerance of the
        # first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        # END_SUB_TUTORIAL

    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        # BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        # Ensuring Collision Updates Are Receieved
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # If the Python node dies before publishing a collision object update message, the message
        # could get lost and the box will not appear. To ensure that the updates are
        # made, we wait until we see the changes reflected in the
        # ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        # For the purpose of this tutorial, we call this function after adding,
        # removing, attaching or detaching an object in the planning scene. We then wait
        # until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        # END_SUB_TUTORIAL

    def add_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        # BEGIN_SUB_TUTORIAL add_box
        ##
        # Adding Objects to the Planning Scene
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # First, we will create a box in the planning scene at the location of the left finger:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "gripper_tip_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.0  # slightly above the end effector
        box_name = "box_name"
        scene.add_box(box_name, box_pose, size=(0.04, 0.04, 0.04))
        # scene.add_mesh(box_name, box_pose, "/root/ros_ws/src/ros-universal-robots/ur3e_dual_moveit_config/meshes/tool-holder-pick.stl")

        # END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def add_box2(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        # BEGIN_SUB_TUTORIAL add_box
        ##
        # Adding Objects to the Planning Scene
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # First, we will create a box in the planning scene at the location of the left finger:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0.93  # slightly above the end effector
        box_pose.pose.position.y = 0.25  # slightly above the end effector
        box_pose.pose.position.z = 1.02  # slightly above the end effector
        box_name = "box_name"
        scene.add_box(box_name, box_pose, size=(0.03, 0.03, 0.03))
        # scene.add_mesh(box_name, box_pose, "/root/ros_ws/src/ros-universal-robots/ur3e_dual_moveit_config/meshes/tool-holder-pick.stl")

        # END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.group_names

        # BEGIN_SUB_TUTORIAL attach_object
        ##
        # Attaching Objects to the Robot
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Next, we will attach the box to the ur3e wrist. Manipulating objects requires the
        # robot be able to touch them without the planning scene reporting the contact as a
        # collision. By adding link names to the ``touch_links`` array, we are telling the
        # planning scene to ignore collisions between those links and the box. For the ur3e
        # robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        # you should change this value to the name of your end effector group name.
        grasping_group = 'gripper'
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        # END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)

    def detach_box(self, name=None, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name if name is None else name
        scene = self.scene
        eef_link = self.eef_link

        # BEGIN_SUB_TUTORIAL detach_object
        ##
        # Detaching Objects from the Robot
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can also detach and remove the object from the planning scene:
        scene.remove_attached_object(eef_link, name=box_name)
        # END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)

    def remove_box(self, name=None, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name if name is None else name
        scene = self.scene

        # BEGIN_SUB_TUTORIAL remove_object
        ##
        # Removing Objects from the Planning Scene
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can remove the box from the world.
        scene.remove_world_object(box_name)

        # **Note:** The object must be detached before we can remove it from the world
        # END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)


def main():
    """Moveit Control Example

    Test several functionalities of MoveIt with the Python interface.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument(
        '-j', '--move-joints', action='store_true', help='Movement using joint state goal')
    parser.add_argument(
        '-p', '--move-pose',   action='store_true', help='Movement using a pose goal')
    parser.add_argument(
        '-a', '--add-box',     action='store_true', help='Add & Attach a box to the gripper')
    parser.add_argument(
        '-r', '--remove-box',  action='store_true', help='Detach & Remove box')
    parser.add_argument(
        '-d', '--detach-box',  action='store_true', help='Detach box')
    parser.add_argument(
        '--tutorial',         action='store_true', help='Execute original tutorial sequence')
    args = parser.parse_args(rospy.myargv()[1:])

    tutorial = MoveGroupPythonIntefaceTutorial()
    rospy.sleep(1)
    if args.move_joints:
        tutorial.display_basic_info()
        tutorial.go_to_joint_state()
    elif args.move_pose:
        tutorial.display_basic_info()
        tutorial.go_to_pose_goal()
    elif args.add_box:
        tutorial.add_box2()
    elif args.remove_box:
        tutorial.remove_box()
    elif args.detach_box:
        tutorial.detach_box()
    elif args.tutorial:
        try:
            print("")
            print("----------------------------------------------------------")
            print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
            print("----------------------------------------------------------")
            print("Press Ctrl-D to exit at any time")
            print("")
            print("============ Press `Enter` to begin the tutorial by setting up the moveit_commander ...")
            input()
            tutorial.display_basic_info()
            print("============ Press `Enter` to execute a movement using a joint state goal ...")
            input()
            tutorial.go_to_joint_state()

            print("============ Press `Enter` to execute a movement using a pose goal ...")
            input()
            tutorial.go_to_pose_goal()

            print("============ Press `Enter` to plan and display a Cartesian path ...")
            input()
            cartesian_plan, fraction = tutorial.plan_cartesian_path()

            print("============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ...")
            input()
            tutorial.display_trajectory(cartesian_plan)

            print("============ Press `Enter` to execute a saved path ...")
            input()
            tutorial.execute_plan(cartesian_plan)

            print("============ Press `Enter` to add a box to the planning scene ...")
            input()
            tutorial.add_box()

            print("============ Press `Enter` to attach a Box to the ur3e robot ...")
            input()
            tutorial.attach_box()

            print("============ Press `Enter` to plan and execute a path with an attached collision object ...")
            input()
            cartesian_plan, fraction = tutorial.plan_cartesian_path(scale=-1)
            tutorial.execute_plan(cartesian_plan)

            print("============ Press `Enter` to detach the box from the ur3e robot ...")
            input()
            tutorial.detach_box()

            print("============ Press `Enter` to remove the box from the planning scene ...")
            input()
            tutorial.remove_box()

            print("============ Python tutorial demo complete!")
        except rospy.ROSInterruptException:
            return
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()

# BEGIN_TUTORIAL
# .. _moveit_commander:
# http://docs.ros.org/melodic/api/moveit_commander/html/namespacemoveit__commander.html
##
# .. _MoveGroupCommander:
# http://docs.ros.org/melodic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html
##
# .. _RobotCommander:
# http://docs.ros.org/melodic/api/moveit_commander/html/classmoveit__commander_1_1robot_1_1RobotCommander.html
##
# .. _PlanningSceneInterface:
# http://docs.ros.org/melodic/api/moveit_commander/html/classmoveit__commander_1_1planning__scene__interface_1_1PlanningSceneInterface.html
##
# .. _DisplayTrajectory:
# http://docs.ros.org/melodic/api/moveit_msgs/html/msg/DisplayTrajectory.html
##
# .. _RobotTrajectory:
# http://docs.ros.org/melodic/api/moveit_msgs/html/msg/RobotTrajectory.html
##
# .. _rospy:
# http://docs.ros.org/melodic/api/rospy/html/
# CALL_SUB_TUTORIAL imports
# CALL_SUB_TUTORIAL setup
# CALL_SUB_TUTORIAL basic_info
# CALL_SUB_TUTORIAL plan_to_joint_state
# CALL_SUB_TUTORIAL plan_to_pose
# CALL_SUB_TUTORIAL plan_cartesian_path
# CALL_SUB_TUTORIAL display_trajectory
# CALL_SUB_TUTORIAL execute_plan
# CALL_SUB_TUTORIAL add_box
# CALL_SUB_TUTORIAL wait_for_scene_update
# CALL_SUB_TUTORIAL attach_object
# CALL_SUB_TUTORIAL detach_object
# CALL_SUB_TUTORIAL remove_object
# END_TUTORIAL
