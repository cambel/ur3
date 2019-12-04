Universal Robot UR3
===

Custom ROS packages for the UR3 Robot with a Robotiq gripper. Tested on Ros Kinetic Ubuntu 16.04. Python 2.7 and 3.6. Preferibly use Python 3.6.

## Installation

This will assume that you already have a catkin workspace 'ros_ws'. Go to the source directory of the workspace
  ```
  $ cd ~/ros_ws/src
  ```

Clone this repo
  ```
  $ git clone https://github.com/cambel/ur3
  ```

Install ros dependencies
  ```
  $ cd ~/ros_ws
  $ rosinstall ~/ros_ws/src /opt/ros/kinetic src/ur3/dependencies.rosinstall
  $ sudo apt-get update
  $ rosdep fix-permissions
  $ rosdep update
  $ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
  ```

Build using catkin build
  ```
  $ cd ~/ros_ws/
  $ catkin clean
  $ catkin build
  ```

## Visualization of UR3 in RViz

To visualize the model of the robot with a gripper, launch the following:
  ```
  $ roslaunch ur3_description display_with_gripper.launch
  ```
You can then use the sliders to change the joint values and the gripper values.

## Simulation in Gazebo 7

To simulate the robot launch the following:
  ```
  $ roslaunch ur3_gazebo ur3_cubes.launch
  ```
or using ur3e:
  ```
  $ roslaunch ur3_gazebo ur3e.launch
  ```

By default the simulation starts paused. Unpause the simulation. You can then send commands to the
joints or to the gripper.

The following is an example of an action client to change the gripper configuration. Open a new
terminal, and then execute:
  ```
  $ rosrun ur3_gazebo send_gripper.py --value 0.5
  ```
where the value is a float between 0.0 (closed) and 0.8 (open).

An example of sending joints values to the robot can be executed as follows:
  ```
  $ rosrun ur3_gazebo send_joints.py
  ```
To change the values of the joints, the file `send_joints.py` must be modified.

An easy way to control the robot using the keyboard can be found in the script:
  ```
  $ rosrun ur_control joint_position_keyboard.py
  ```
Press SPACE to get a list of all valid commands to control either each independent joint or the end effector position x,y,z and rotations.

Another option of easy control is using `rqt`
