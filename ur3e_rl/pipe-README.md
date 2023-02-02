### Initialize camera sensor
roslaunch realsense2_camera rs_camera.launch color_width:=1920 color_height:=1080 color_fps:=15

### bring up robots
roslaunch ur3e_dual_control ur3e_dual_bringup.launch

### start moveit planner
roslaunch ur3e_dual_moveit_config start_real_dual_ur3e_moveit.launch

