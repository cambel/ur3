Universal Robot UR3/UR3e
===
<img src="https://github.com/cambel/ur3/blob/master/wiki/ur3e.gif?raw=true" alt="UR3e & Robotiq Hand-e" width="250"><img src="https://github.com/cambel/ur3/blob/master/wiki/ur3.gif?raw=true" alt="UR3 & Robotiq 85" width="250">


Custom ROS packages for the UR3 Robot with a gripper Robotiq 85 and the UR3e robot with a gripper Robotiq Hand-e. 
Tested on Ros Kinetic Ubuntu 16.04. Python 2.7 and 3.6. Preferibly use Python 3.6.
Also tested on Ros Melodic Ubuntu 18.04. Python 2.7 and 3.6. Preferibly use Python 3.6.

For ROS Melodic Ubuntu 18.04. Two dependencies repositories has to be updated: **gazebo_ros_link_attacher** and **robotiq**. Both git repositories have a "melodic-devel" branch. Moving those 2 repositories to the melodic-devel branch makes this repository compatible with ROS Melodic.

## Installation 

### from docker
install basic dependencies and tools 
  ```
  $ sudo apt-get install git htop vim curl bindfs
  ```

Install docker 

**Ubuntu 16.04 or 18.04**
  ```
  $ sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
  $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu  $(lsb_release -cs)  stable" 
  $ sudo apt-get update
  $ sudo apt-get install docker-ce
  ```

Fix docker permissions
  ```
  $ sudo groupadd docker
  $ sudo usermod -aG docker $USER
  $ newgrp docker 
  ```
may required log out to recognize permissions

Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  ```
  $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
  $ sudo systemctl restart docker
  ```

Clone this repo
  ```
  $ git clone https://github.com/cambel/ur3
  ```

Build the dockerfile image
  ```
  $ cd ur3
  $ docker build -t ros-ur3 .
  ```

If everything builds right, use the launch_docker.sh script to start the environment

  ```
  bash launch_docker.sh
  ```

Then enter the docker environment on each new terminal using the command
  ```
  docker exec -it ros-ur3 bash
  ```

### compile source (this repo)

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

for ROS Melodic only  
  ```
  $ cd ~/ros_ws/src/gazebo_ros_link_attacher
  $ git checkout melodic-devel
  $ cd ~/ros_ws/src/robotiq
  $ git fetch origin && git checkout melodic-devel"
  ```

Build using catkin build
  ```
  $ cd ~/ros_ws/
  $ catkin clean
  $ catkin build
  ```

## Examples

### Visualization of UR3 in RViz

To visualize the model of the robot with a gripper, launch the following:
  ```
  $ roslaunch ur3_description display_with_gripper.launch
  ```
You can then use the sliders to change the joint values and the gripper values.

### Simulation in Gazebo 7/9
<img src="https://github.com/cambel/ur3/blob/master/wiki/ur3-e.png?raw=true" width="500">
<!-- ![ur3/ur3e gazebo simulator](https://github.com/cambel/ur3/blob/master/wiki/ur3-e.png?raw=true) -->

To simulate the robot launch the following:
  ```
  $ roslaunch ur3_gazebo ur3_cubes.launch grasp_plugin:=1
  ```
or using ur3e:
  ```
  $ roslaunch ur3_gazebo ur3e_cubes.launch grasp_plugin:=1
  ```

By default the simulation starts paused. Unpause the simulation. You can then send commands to the
joints or to the gripper.

An example of sending joints values to the robot can be executed as follows:
  ```
  $ rosrun ur_control sim_controller_examples.py -m
  ```
To change the values of the joints, the file `sim_controller_examples.py` must be modified.

Similarly, the script include examples to control the robot's end-effector position, gripper and an example of performing grasping.
Execute the following command to see the available examples.
  ```
  $ rosrun ur_control sim_controller_examples.py --help
  ```

An easy way to control the robot using the keyboard can be found in the script:
  ```
  $ rosrun ur_control joint_position_keyboard.py
  ```
Press SPACE to get a list of all valid commands to control either each independent joint or the end effector position x,y,z and rotations.
To have access to the gripper controller include the option `--gripper`

Another option of easy control is using `rqt`
