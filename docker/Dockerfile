# Docker for ur3 repo
# ros-noetic-base, gazebo11, python3 libraries 
# Python3 3 version

FROM osrf/ros:noetic-desktop-full
LABEL maintainer Cristian Beltran "beltran@hlab.sys.es.osaka-u.ac.jp"

ENV ROS_DISTRO noetic

# install universal robot ros packages
RUN apt-get update && apt-get install -y \
    # ROS utils 
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-rqt-common-plugins \
    ros-$ROS_DISTRO-rqt-joint-trajectory-controller \
    ros-$ROS_DISTRO-rqt-tf-tree \
    ros-$ROS_DISTRO-rqt-multiplot \
    ros-$ROS_DISTRO-moveit \
    # install catkin
    ros-$ROS_DISTRO-catkin \
    python3-catkin-tools \
    # utils
    git \
    locate \
    aptitude \
    vim htop \
    curl wget \
    spacenavd \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

### Ros Workspace ###
# Set up the workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && mkdir -p ~/ros_ws/src \
               && cd ~/ros_ws/src \
               && git clone https://github.com/cambel/ur3.git ros-ur -b noetic-devel"

# Updating ROSDEP and installing dependencies
RUN cd ~/ros_ws \
    && rosinstall ~/ros_ws/src /opt/ros/$ROS_DISTRO src/ros-ur/dependencies.rosinstall \
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
# RUN /bin/bash -c "cd ~/ \
#                && mkdir pylibs && cd pylibs \
#                && git clone https://github.com/cambel/ur_ikfast.git \
#                && cd ur_ikfast && pip install -e ."

# Download gazebo models only once
RUN mkdir -p ~/.gazebo
RUN git clone https://github.com/osrf/gazebo_models ~/.gazebo/models

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# setup environment
EXPOSE 11345

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'source ~/ros_ws/devel/setup.bash' >> ~/.bashrc

RUN echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
RUN echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ros_ws/src/ros-universal-robots/ur_gripper_gazebo/models/' >> ~/.bashrc
