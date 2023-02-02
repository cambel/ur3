#!/usr/bin/env python
import rospy
from ur3e_openai.common import load_environment

from drl.agents.actor_critic_agents.SAC import SAC
from drl.agents.actor_critic_agents.TD3 import TD3

from drl.utilities.data_structures.Config import Config
from drl.agents.Trainer import Trainer

import sys
import signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

rospy.init_node('ur3e_hybrid',
                anonymous=True,
                log_level=rospy.ERROR)


env = load_environment('UR3eHybridControlEnv-v0',
                       max_episode_steps=150)

# load parameters
num_episodes_to_run = rospy.get_param("ur3e/num_episodes_to_run")
discount_rate = rospy.get_param("ur3e/discount_rate")
min_steps_before_learning = rospy.get_param("ur3e/min_steps_before_learning")
batch_size = rospy.get_param("ur3e/batch_size")

config = Config()
config.seed = 1
config.environment = env
config.num_episodes_to_run = num_episodes_to_run
config.file_to_save_data_results = "UR3e_Results_Data.pkl"
config.file_to_save_results_graph = "UR3e_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

config.hyperparameters = {
    "Actor_Critic_Agents": {
        "learning_rate": 0.005,
        "linear_hidden_units": [64, 64],
        "gradient_clipping_norm": 5.0,
        "discount_rate": discount_rate,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },
        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 30000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },
        "min_steps_before_learning": min_steps_before_learning,
        "batch_size": batch_size,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "update_every_n_steps": 100,
        "learning_updates_per_learning_session": 5,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == '__main__':
    AGENTS = [SAC]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
