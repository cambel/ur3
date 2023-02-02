# Tutorial for using RL with UR3e + force control on simulation


## Peg-in-hole example environment

<img src="https://drive.google.com/uc?export=view&id=1AzydhTNjup9C83oEjI0JmvaGGZCeKPnb" width="400">

1. Start the gazebo peg-in-hole environment 
    ```
    roslaunch ur3_gazebo ur3e_dual.launch peg_shape:=cube
    ```
2. Unpause the environment
3. Execute trainer script
   ```
   rosrun ur3e_rl tf2rl_sac.py --env-id 1
   ```
   This script will start a peg-in-hole training using domain randomization where the initial position of the robot, the position of the task board and the stiffness of the task board will change every few episodes. The configuration file for this training environment can be found in ur3e_rl/config/simulation/peg_in_hole.yaml
   The training session will last 5000 steps as define in the parameters of the tf2rl_sac.py script where each episode has at most 300 steps as defines in the peg_in_hole.yaml
   The trainer script support several parameters that can be seen with the command
   ```
   rosrun ur3e_rl tf2rl_sac.py --help
   ```

## Testing the learned policy
The learned policy will be safe in the folder (current folder)/results/time_env_name/*
The learned policy can be executed using the following command
```
rosrun ur3e_rl evaluate_policy.py results/time_env_name -n 2
```
the parameter -n is the number of trials to be executed. The policy is tested using the same parameters that were defined during training but that are saved to the file ros_gym_env_params.yaml located inside the policy folder.

# Tutorial for dual environment / pipe coating task

1. Start the gazebo environment 

    `roslaunch ur3e_dual_gazebo dual_ur3e.launch`

   Or the real robot environment

    `roslaunch ur3e_dual_control ur3e_dual_bringup.launch`

2. Start the moveit planner
   
   sim

    `roslaunch ur3e_dual_moveit_config start_sim_dual_ur3e_moveit.launch`
  
   real

    `roslaunch ur3e_dual_moveit_config start_real_dual_ur3e_moveit.launch`

3. Start the moveit-planner service

    `rosrun ur3e_dual_control handle_moveit_service.py`

4. Execute the policy/training script

    `rosrun ur3e_rl simple_env.py -e6 -a1`

# 2023-01 Executing peg in hole tasks in the hanging position

1. Run this gazebo environment:

    `roslaunch ur3e_dual_gazebo single_ur3e_simple_gripper.launch`

2. Run config file 17:

    `rosrun ur3e_rl tf2rl_sac.py -e 17`
