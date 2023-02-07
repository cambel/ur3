#!/usr/bin/python3
# Example of adding/removing models to gazebo simulator
import argparse
import rospy
import numpy as np
import copy
from ur_control import transformations
from ur_control.arm import Arm

from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.model import Model
from ur_gazebo.basic_models import SPHERE, PEG_BOARD, BOX, SPHERE_COLLISION, get_box_model, get_button_model, get_peg_board_model

rospy.init_node('gazebo_spawner_ur3e')
spawner = GazeboModels('ur3_gazebo')


def place_target():
    sphere = SPHERE % ("target", "0.02", "GreenTransparent")
    model_names = ["target"]
    objpose = [[-0.13101034,  0.37818616,  0.50852045, 0, 0, 0]]

    models = [Model(model_names[0], objpose[0], file_type='string', string_model=sphere, reference_frame="world")]
    spawner.load_models(models)


def place_ball():
    sphere = SPHERE_COLLISION.format("ball", "0.1", "Yellow", "0.1", 2e5)
    model_names = ["ball"]
    objpose = [[0.608678,  0.0,  (0.25939 + 0.685)], [0, 0, 0]]
    #  0.25939 + 0.685
    models = [Model(model_names[0], objpose[0], file_type='string', string_model=sphere, reference_frame="world")]
    spawner.load_models(models)


def place_cube():
    cube_lenght = "0.2"
    obj = BOX % ("box", cube_lenght, cube_lenght, cube_lenght, "Yellow", cube_lenght, cube_lenght, cube_lenght)
    model_names = ["box"]
    objpose = [[0.618678,  0.0,  0.955148], [0, 0.0, 0, 0.0]]

    models = [Model(model_names[0], objpose[0], file_type='string', string_model=obj, reference_frame="world")]
    spawner.load_models(models)

def place_models():
    model_names = ["multi_peg_board"]
    model_names = ["simple_peg_board"]
    objpose = [[-0.45, -0.20, 0.86], [0, 0.1986693, 0, 0.9800666]]
    models = [Model(model_names[0], objpose[0], orientation=objpose[1])]
    spawner.load_models(models)


def place_soft():
    name = "simple_peg_board"
    
    objpose = [-0.10, 0.35, 0.10, 3.1415, 0, 0.0] # hanging pose
    # objpose = [0.17021, -0.36557, 0.35, 0, 0, 0.0] # hanging pose

    stiffness = 1e6
    string_model = get_peg_board_model(kp=stiffness, mu=1, mu2=1, peg_shape="cylinder", color=[0,0.8,0,0.1])
    models = [Model(name, pose=objpose, file_type='string', string_model=string_model, reference_frame="base_link")]
    spawner.load_models(models)

def place_collision_cube():
    size = 0.02
    string_model1 = get_box_model("cube", [size/2.,size/2.,size/2.], size, mu=2, mu2=2, color="Green", mass=0.5)
    string_model2 = get_box_model("cube", [size/2.,size/2.,size/2.], size, color="Blue", mass=1)
    string_model3 = get_box_model("cube", [size/2.,size/2.,size/2.], size, color="Red", mass=2)
    string_model4 = get_box_model("cube", [size/2.,size/2.,size/2.], size, mu=2, mu2=2, color="DarkRed", mass=10)
    string_models = [
        get_box_model("cube", [size/2.,size/2.,size/2.], size, mu=0.1, mu2=0.1, color="Green", mass=0.1),
        # get_box_model("cube", [size/2.,size/2.,size/2.], size, color="Blue", mass=1),
        # get_box_model("cube", [size/2.,size/2.,size/2.], size, color="Red", mass=2),
        # get_box_model("cube", [size/2.,size/2.,size/2.], size, mu=2, mu2=2, color="DarkRed", mass=10),
    ]
    initial_pose = [0.36,0.10,0.69,0,0,0]
    models = []
    for i in range(len(string_models)):
        pose = copy.copy(initial_pose)
        pose[1] -= 0.05*i
        model = Model("cube", pose, file_type="string", string_model=string_models[i], model_id="cube%s" % i)
        models.append(model)
    spawner.load_models(models)

def place_door():
    name = "hinged_door"
    objpose = [[-0.40, 0.20, 0.76], [0, 0, 0.9238795, 0.3826834]]
    models = [[Model(name, objpose[0], orientation=objpose[1])]]
    spawner.load_models(models)

def place_eef():
    arm = Arm(ft_topic='wrench',
              gripper=False, namespace="",
              joint_names_prefix="",
              robot_urdf="ur3e",
              ee_link="gripper_tip_link")
    sphere = SPHERE.format("ball", 0.0025, "GreenTransparent")
    model_names = ["eef"]
    models = [Model(model_names[0], arm.end_effector(), 
              file_type='string',
              string_model=sphere, reference_frame="base_link")]
    spawner.load_models(models)

def place_aruco():
    name = "Apriltag36_11_00000"
    # objpose = [[-0.53, 0.25, 0.82], [0,0,0]] # world coordinates
    objpose = [[0.20, -0.30, 1.00], [0, 0, 0]]  # world coordinates
    objpose = [[-0.65, 0.0, 0.78], [0, 0, np.pi/8.0]]  # world coordinates
    models = [Model(name, objpose[0], orientation=objpose[1], file_type='sdf', reference_frame="world")]
    spawner.load_models(models)

def place_button():
    name = "button"
    initial_pose = [0.46, 0.07, 0.72, 0, 0, 0]  # world coordinates
    # models = [Model(name, pose=objpose, file_type='sdf', reference_frame="world")]
    string_models = [
        get_button_model(color=[0,1,0,0], damping=0., friction=0., spring_stiffness=-100.),
        get_button_model(color=[0,0,1,0],  damping=1., friction=1., spring_stiffness=-300.),
        get_button_model(color=[1,0,0,0],  damping=10., friction=10., spring_stiffness=-500.),
    ]
    models = []
    for i in range(len(string_models)):
        pose = copy.copy(initial_pose)
        pose[1] -= 0.075*i
        model = Model("button", pose, file_type="string", string_model=string_models[i], model_id="button_%s" % i)
        models.append(model)
    spawner.load_models(models)

def place_peg():
    arm = Arm(ft_topic='wrench',
            gripper=False, namespace="",
            joint_names_prefix="",
            robot_urdf="ur3e",
            ee_link="gripper_tip_link")
    q = [1.4308, -1.103, 1.4793, -1.9346, -1.5709, -0.1394]
    arm.set_joint_positions(q, wait=True, t=1)
    name = "peg_cube"
    initial_pose = [0.40, 0.07, 0.72, 0, 0, 0]  # world coordinates
    models = [Model(name, pose=initial_pose, file_type='sdf', reference_frame="world")]
    spawner.load_models(models)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--place', action='store_true',
                        help='Place models')
    parser.add_argument('--target', action='store_true',
                        help='Place targets')
    parser.add_argument('--soft', action='store_true',
                        help='Place soft peg board')
    parser.add_argument('--door', action='store_true',
                        help='Place door')
    parser.add_argument('--aruco', action='store_true',
                        help='Place aruco marker')
    parser.add_argument('--ball', action='store_true',
                        help='Place ball with collision')
    parser.add_argument('--cube', action='store_true',
                        help='Place cube with collision')
    parser.add_argument('--button', action='store_true',
                        help='Place button with collision')
    parser.add_argument('--eef', action='store_true',
                        help='Place button with collision')
    parser.add_argument('--peg', action='store_true',
                        help='Place button with collision')
    args = parser.parse_args()


    if args.place:
        place_models()
    if args.target:
        place_target()
    if args.soft:
        place_soft()
    if args.door:
        place_door()
    if args.aruco:
        place_aruco()
    if args.ball:
        place_ball()
    if args.cube:
        # place_cube()
        place_collision_cube()
    if args.button:
        place_button()
    if args.eef:
        place_eef()
    if args.peg:
        place_peg()

if __name__ == "__main__":
    main()
