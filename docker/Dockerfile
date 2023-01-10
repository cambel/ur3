# Docker for ur3 repo
# ros-noetic-base, gazebo11, python3 libraries 
# Python3 3 version

FROM osrf/ros:noetic-desktop-full
LABEL maintainer Cristian Beltran "beltran@hlab.sys.es.osaka-u.ac.jp"

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# install universal robot ros packages
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-rqt-common-plugins \
    # install catkin
    ros-$ROS_DISTRO-catkin \
    python3-catkin-tools \
    # Install Numpy Boost
    libboost-dev \
    libboost-python-dev \
    libboost-system-dev \
    libboost-all-dev \
    libatlas-base-dev \
    libprotobuf-dev \
    protobuf-compiler \
    # python3 dependencies
    python3-setuptools \
    python3-tk \
    python3-numpy \
    # utils
    locate \
    aptitude \
    vim htop tmux \
    curl wget \
    tk \
    spacenavd \
    && rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install -y libignition-common3-graphics python3-catkin-lint python3-pip git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install osrf-pycommon

### Ros Workspace ###
# Set up the workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && mkdir -p ~/ros_ws/src \
               && cd ~/ros_ws/ \
               && catkin config -init --blacklist python_orocos_kdl orocos_kinematics_dynamics orocos_kdl"

# Installing repo required for homework
RUN /bin/bash -c "cd ~/ros_ws/src \
               && git clone https://github.com/cambel/ur3.git ros_ur3 -b noetic-devel\
               && git clone https://bitbucket.org/traclabs/trac_ik.git \
               && git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git universal_robots_ros_driver \
               && git clone -b calibration_devel https://github.com/fmauch/universal_robot.git universal_robot"

# Updating ROSDEP and installing dependencies
RUN cd ~/ros_ws \
    && rosinstall ~/ros_ws/src /opt/ros/$ROS_DISTRO src/ros_ur3/dependencies.rosinstall \
    && apt-get update \
    && rosdep fix-permissions \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING UTF-8

## Python3 libraries ##
RUN python3 -m pip install pip --upgrade && \
    pip install matplotlib==2.2.3 spicy protobuf pyyaml pyquaternion rospkg \
    lxml tqdm catkin-pkg empy PyVirtualDisplay defusedxml gym psutil pyprind

# # Compiling ros workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && cd ~/ros_ws/ \
               && rm -rf build \
               && catkin build"

################################################
# Custom python3 libs
################################################
# ur_ikfast
RUN /bin/bash -c "cd ~/ \
               && mkdir pylibs && cd pylibs \
               && git clone https://github.com/cambel/ur_ikfast.git \
               && cd ur_ikfast && pip install -e ."

RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-rqt-joint-trajectory-controller \
    ros-$ROS_DISTRO-rqt-tf-tree \
    ros-$ROS_DISTRO-rqt-multiplot \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-moveit \
    ros-$ROS_DISTRO-moveit-runtime \
    ros-$ROS_DISTRO-moveit-resources \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && apt-get install -y \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Download gazebo models only once
RUN mkdir -p ~/.gazebo
RUN git clone https://github.com/osrf/gazebo_models ~/.gazebo/models

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# setup environment
EXPOSE 11345

RUN echo "alias python=/bin/python3" >> ~/.bashrc

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'source ~/ros_ws/devel/setup.bash' >> ~/.bashrc

RUN echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
RUN echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ros_ws/src/ros-universal-robots/ur3_gazebo/models/' >> ~/.bashrc
RUN echo "alias ur3e_dual_sim='roslaunch ur3_gazebo ur3e_dual_gripper.launch pause:=0'" >> ~/.bashrc
RUN echo "alias ur3e_sim='roslaunch ur3_gazebo ur3e_cubes.launch'" >> ~/.bashrc
RUN echo 'function ur3e_robot { roslaunch ur_robot_driver ur3e_bringup.launch robot_ip:="$1" kinematics_config:=/root/dev/tools/my_robot_calibration.yaml; }'  >> ~/.bashrc
RUN echo 'function ur3_robot { roslaunch ur_robot_driver ur3_bringup.launch robot_ip:="$1" kinematics_config:=/root/dev/tools/my_robot_calibration.yaml; }'  >> ~/.bashrc
