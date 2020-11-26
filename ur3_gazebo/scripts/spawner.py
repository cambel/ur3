#!/usr/bin/python3
# Example of adding/removing models to gazebo simulator
import argparse
import rospy
import numpy as np
import copy

from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.model import Model
from ur_gazebo.basic_models import SPHERE, PEG_BOARD, BOX, SPHERE_COLLISION

rospy.init_node('gazebo_spawner_ur3e')
spawner = GazeboModels('ur3_gazebo')


def place_target():
    sphere = SPHERE % ("target", "0.02", "GreenTransparent")
    model_names = ["target"]
    objpose = [[0.0131,  0.4019,  0.3026]]
    objpose = [[-0.13101034,  0.37818616,  0.50852045], [0, 0, 0]]

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
    objpose = [[-0.3349516,  0.00327044,  0.45290458], [-0.50128434, -0.49779569,  0.50398595,  0.496]]

    stiffness = 1e5
    g = 0
    if stiffness > 1e5:
        g = np.interp(stiffness, [1e5, 1e6], [1, 0])
    else:
        g = np.interp(stiffness, [1e4, 1e5], [0, 1])
    b = np.interp(stiffness, [1e4, 1e5], [1, 0])
    r = np.interp(stiffness, [1e5, 1e6], [0, 1])
    string_model = PEG_BOARD.format(r, g, b, stiffness)
    models = [[Model(name, objpose[0], orientation=objpose[1], file_type='string', string_model=string_model, reference_frame="base_link")]]
    spawner.load_models(models)


def place_door():
    name = "hinged_door"
    objpose = [[-0.40, 0.20, 0.76], [0, 0, 0.9238795, 0.3826834]]
    models = [[Model(name, objpose[0], orientation=objpose[1])]]
    spawner.load_models(models)


def place_aruco():
    name = "Apriltag36_11_00000"
    # objpose = [[-0.53, 0.25, 0.82], [0,0,0]] # world coordinates
    objpose = [[0.20, -0.30, 1.00], [0, 0, 0]]  # world coordinates
    objpose = [[-0.65, 0.0, 0.78], [0, 0, np.pi/8.0]]  # world coordinates
    models = [Model(name, objpose[0], orientation=objpose[1], file_type='sdf', reference_frame="world")]
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
        place_cube()

if __name__ == "__main__":
    main()
