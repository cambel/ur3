docker stop ros-ur3
docker create -it \
    --gpus all \
    --runtime=nvidia --rm -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --privileged \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev:/dev \
    -v ros_ws_container:/root/ros_ws/src/ros_ur3 \
    --network host \
    -w '/root/dev/' \
    --name=ros-ur3 \
    ros-ur3 \
    && export containerId=$(docker ps -l -q) \
    && xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId` \
    && docker start $containerId

FILE=~/dev/container_ws/README.md
if [ ! -f "$FILE" ]; then
    sudo bindfs --map=root/$USER /var/lib/docker/volumes/ros_ws_container/_data ~/dev/container_ws
fi