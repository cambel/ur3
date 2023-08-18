#!/usr/bin/python3
import sys
import signal
from datetime import datetime
from ur_control import conversions
from ur_control.fzi_cartesian_compliance_controller import CompliantController
import rospy
import numpy as np

from matplotlib import pyplot as plt

from ur_gazebo.basic_models import get_button_model
from ur_gazebo.model import Model
from ur_gazebo.gazebo_spawner import GazeboModels

import moveit_commander
from o2ac_routines.robot_base import RobotBase
import tf

import optuna


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

pi = np.pi


def slice_vegetable_at_fix_velocity(b_bot: RobotBase, arm: CompliantController, displacement=0.03):

    slicing_pose = conversions.to_pose_stamped("cutting_board_surface", [0.03, 0.0, 0.065, -pi, pi/2, 0])
    b_bot.go_to_pose_goal(slicing_pose, speed=0.2, end_effector_link="b_bot_knife_center", move_lin=True)

    # move_down(self, num_of_points=10, distance=0.05, velocity=0.001)  # 5cm at 1 mm/s

    speed = 0.02

    target_pose = b_bot.move_lin_rel(relative_translation=[0.0, 0, -0.032], pose_only=True)
    target_pose = listener.transformPose("world", target_pose).pose
    b_bot.robot_group.limit_max_cartesian_link_speed(speed=speed, link_name="b_bot_gripper_tip_link")
    trajectory, _ = b_bot.robot_group.compute_cartesian_path([target_pose], eef_step=0.001, jump_threshold=3.0)
    # print("res", type(trajectory), trajectory)
    b_bot.robot_group.clear_max_cartesian_link_speed()
    arm.zero_ft_sensor()
    b_bot.robot_group.execute(trajectory, wait=False)

    duration = displacement/speed + 0.5
    record_force_profile(arm, duration=duration, plot=True)

    b_bot.go_to_pose_goal(slicing_pose, speed=1.0, end_effector_link="b_bot_knife_center", move_lin=True)


def record_force_profile(robot: CompliantController, duration=5, plot=False):

    ft_sensor_data = []

    # robot._init_ik_solver(base_link=robot.base_link, ee_link="b_bot_tool0")
    robot.set_ft_filtering(active=True)
    init_time = rospy.get_time()
    r = rospy.Rate(500)
    while rospy.get_time() - init_time < duration:
        ft_sensor_data.append(robot.get_ee_wrench().tolist() + [rospy.get_time() - init_time])
        r.sleep()
    rospy.loginfo(" =====  Finished ==== ")
    robot._init_ik_solver(base_link=robot.base_link, ee_link=robot.ee_link)

    if plot:
        time = np.linspace(0, duration, len(ft_sensor_data))
        for i, ax in enumerate(["X", "Y", "Z"]):
            plt.plot(time, np.array(ft_sensor_data)[:, i], label=ax)
        plt.ylim((-35, 20))
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Force N")
        plt.legend()
        plt.show()

    return np.linalg.norm(np.array(ft_sensor_data)[:, :3], axis=1)


def slice_button(arm: CompliantController, erp, cfm,):
    button_initial_pose = [-0.12, 0.02, 0.875, 0, 0, 1.5707]
    # go to an initial position
    init_q = [1.35233835, -1.54644853, 1.80239741, -1.82021819, -1.56438015, 1.34947447]
    arm.set_joint_positions(init_q, t=0.5, wait=True)
    # arm.move_relative([0,0,-0.016,0,0,0], wait=True, duration=2, relative_to_tcp=False)
    # print(arm.joint_angles())
    # Spawn the button with some stiffness parameter
    string_model = get_button_model(erp=erp, cfm=cfm, base_mass=1.)
    box_model = Model("block", button_initial_pose, file_type="string", string_model=string_model, model_id="target_block", reference_frame="o2ac_ground")
    spawner.reset_model(box_model)
    # move at a constant speed up to the button base
    # record force data
    arm.zero_ft_sensor()
    # arm.move_relative([0, 0, -0.040, 0, 0, 0], wait=False, duration=1, relative_to_tcp=False)
    duration = 0.03/0.02 + 1.0
    ft_data = record_force_profile(arm, duration=1.1, plot=False)
    slice_vegetable_at_fix_velocity(b_bot, arm)
    return max(ft_data)

def plot3d(data):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for d in data:
        ax.scatter(d[0], d[1], d[2])

    ax.set_xlabel('ERP')
    ax.set_ylabel('CFM')
    ax.set_zlabel('Max Force')

    plt.show()

if __name__ == '__main__':
    rospy.init_node('gazebo_spawner')

    moveit_commander.roscpp_initialize(sys.argv)
    moveit_commander.RobotCommander()

    listener = tf.TransformListener()
    b_bot = RobotBase("b_bot", listener)

    spawner = GazeboModels('ur3_gazebo')

    arm = CompliantController(namespace='b_bot',
                              joint_names_prefix='b_bot_',
                              ee_link="knife_center",
                              ft_topic='wrench')

    cucumber_max_ft = 12.03  # cucumber max force 12.036365647719114
    potato_max_ft = 27.49  # potato max force 27.488895519611752
    tomato_max_ft = 30.59  # tomato max force 30.595279872956606

    def objective(trial):
        erp = trial.suggest_float('erp', 0.1, 3.0)
        cfm = trial.suggest_float('cfm', 0.1, 3.0)

        sim_max_ft = slice_button(arm, erp, cfm)

        return (tomato_max_ft - max(sim_max_ft))**2

    # data = []
    # X,Y = np.mgrid[0.15:2.1:0.1, 0.1:3.01:0.15]
    # xy = np.vstack((X.flatten(), Y.flatten())).T

    # for erp, cfm in xy:
    #     max_ft = slice_button(arm, erp, cfm)
    #     data.append([erp, cfm, max_ft])

    # filename = "/root/o2ac-ur/results/gz_btn_data.npy"
    # np.save(filename, data)
    # plot3d(np.array(data))

    # study = optuna.create_study()
    # study.optimize(objective, n_trials=100)
    # print(study.best_params)

    sim_max_ft = slice_button(arm, erp=0.6487128967875418, cfm=1.224723019173888)
    print(sim_max_ft)
    sim_max_ft = slice_button(arm, erp=1.9598064940234345, cfm=1.6929123153209162)
    print(sim_max_ft)
    sim_max_ft = slice_button(arm, erp=0.5545132411475631, cfm=0.429713214268383)
    print(sim_max_ft)

# TODO do some sort of

# cucumber
# Trial 84 finished with value: 0.004822883128000325 and parameters:
# {'erp': 0.6487128967875418, 'cfm': 1.224723019173888}. Best is trial 84 with value: 0.004822883128000325.
# [INFO] [1692327438.612830, 8208.427000]:  =====  Finished ====
# potato
# {'erp': 1.9598064940234345, 'cfm': 1.6929123153209162}
# 27.490900882660867
# [I 2023-08-18 03:06:04,039] Trial 66 finished with value: 8.115895686535418e-07 and parameters: {'erp': 1.9598064940234345, 'cfm': 1.6929123153209162}. Best is trial 66 with value: 8.115895686535418e-07.
# [INFO] [1692327967.249564, 8734.665000]:  =====  Finished ====
# tomato
# {'erp': 0.5545132411475631, 'cfm': 0.429713214268383}
