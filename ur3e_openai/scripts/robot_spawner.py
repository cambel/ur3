#!/usr/bin/python2
import roslaunch
import rospy
import rosparam
import rospkg
import yaml

from os.path import exists
from controller_manager_msgs.srv import *
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteModel
from ur3e_openai.controllers_connection import ControllersConnection

rospack = rospkg.RosPack()

def spawn_robot_model(peg_shape='cube'):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.scriptapi.ROSLaunch()
    ur3_gazebo_path = rospack.get_path("ur3_gazebo")
    launch_filepath = ur3_gazebo_path + "/launch/ur_peg_alone.launch"
    cli_args = [launch_filepath, 'peg_shape:=%s' % peg_shape ]
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    launch.parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    launch.start()
    return launch

#   <node name="robot_controllers" pkg="controller_manager" type="spawner" respawn="false"
        # output="$(arg DEBUG)"
        # args="joint_state_controller scaled_pos_joint_traj_controller $(arg gripper_controller)"/>

def spawn_controllers():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.scriptapi.ROSLaunch()
    ur_control_path = rospack.get_path("ur_control")
    launch_filepath = ur_control_path + "/launch/ur_e_controllers.launch"
    cli_args = [launch_filepath, 'joint_state_controller scaled_pos_joint_traj_controller']
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    launch.parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    launch.start()
    return launch

# def stop_controllers():

# def _stop_controller(self, name):
#     strict = SwitchControllerRequest.STRICT
#     req = SwitchControllerRequest(start_controllers=[],
#                                     stop_controllers=[name],
#                                     strictness=strict)
#     self._switch_srv.call(req)

if __name__ == '__main__':
    rospy.init_node('gazebo_spawner')
    
    unpause_serv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause_serv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    cc = ControllersConnection()
    controllers_list = ['joint_state_controller', 'scaled_pos_joint_traj_controller']
    launch = spawn_robot_model()
    # con = spawn_controllers()
    print(vars(launch))
    print('sleep')
    print('Press ENTER to start controllers')
    raw_input()
    print('wake up')
    cc.load_controllers(controllers_list)
    cc.switch_controllers(controllers_on=controllers_list,
                          controllers_off=[])

    # load_srv = rospy.ServiceProxy('/controller_manager/load_controller', LoadController, persistent=True)
    # unload_srv = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController, persistent=True)
    # switch_srv = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController, persistent=True)
    # list_controllers_service = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers, persistent=True)

    # list_controllers_service.wait_for_service()
    # load_srv.wait_for_service()
    # r= load_srv.call('joint_state_controller')
    # print('load 1', r)
    # print('list', list_controllers_service.call())
    # r=load_srv.call('scaled_pos_joint_traj_controller')
    # print('load 3', r)
    # res = list_controllers_service.call()
    # print('list', res)
    # for i in range(len(res.controller)):
    #     print('?', res.controller[i].name)

    # req = SwitchControllerRequest(start_controllers=controllers_list, 
    #                               stop_controllers=[],
    #                               strictness=SwitchControllerRequest.STRICT)
    # switch_srv.wait_for_service()
    # r=switch_srv.call(req)
    # print('switch', r)
    print('Press ENTER to stop controllers')
    raw_input()

    cc.switch_controllers(controllers_off=controllers_list,
                          controllers_on=[])
    cc.unload_controllers(controllers_list)
    # switch_srv.wait_for_service()
    # req = SwitchControllerRequest(start_controllers=[], 
    #                               stop_controllers=controllers_list,
    #                               strictness=SwitchControllerRequest.STRICT)
    # switch_srv.call(req)
    # con.stop()
    launch.stop()
    launch.parent.shutdown()
    print('Press ENTER to remove')
    raw_input()
    delete_model_srv('robot')
    pause_serv.wait_for_service()
    pause_serv()
    print('Press ENTER to spawn again')
    raw_input()
    try:
        launch = spawn_robot_model('cylinder')
    except:
        pass
    print('Press ENTER to start controllers')
    raw_input()
    cc.load_controllers(controllers_list)
    cc.switch_controllers(controllers_on=controllers_list,
                        controllers_off=[])
    # spawn_controllers()
    # load_srv.call(LoadControllerRequest(name=name))
    # load_srv.call(LoadControllerRequest(name=name))
    # req = SwitchControllerRequest(start_controllers=controllers_list, 
    #                               stop_controllers=[],
    #                               strictness=SwitchControllerRequest.STRICT)
    # switch_srv.wait_for_service()
    # switch_srv.call(req)
    
    print('sleep2')
    # rospy.sleep(10)
    print('wake up')
    # launch.stop()

    try:
        rospy.spin()
    finally:
    # After Ctrl+C, stop all nodes from running
        launch.stop()


    # cli_args = ['/home/mosaic/catkin_ws/src/robot/launch/id.launch','vel:=2.19']
    # roslaunch_args = cli_args[1:]
    # roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

    # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)