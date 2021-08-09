docker stop ros-ur3-melodic
docker create -it \
    --gpus all \
    --runtime=nvidia --rm -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --privileged \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev:/dev \
    -v ~/temp/ur3:/root/ros_ws/src/ros_ur3 \
    --network host \
    -w '/root/' \
    --name=ros-ur3-melodic \
    ros-ur3:melodic \
    && export containerId=$(docker ps -l -q) \
    && xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId` \
    && docker start $containerId