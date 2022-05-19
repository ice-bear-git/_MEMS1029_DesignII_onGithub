import torch
import time
import pybullet as p
import numpy as np

import gym
import sys
import torch

import torch
import time
import pybullet as p
import numpy as np
import argparse

from envs.pybullet_PPO_snake import PPO_Basic_Snake
from policies.PPO import PPO
from policies.network import FeedForwardNN
from policies.eval_policy import eval_policy


class Snake_PPO_V1(PPO_Basic_Snake):
    def __init__(self, target_position=None, action_Scaling_list=None):
        if target_position is None:
            target_position = [1, 0, 0.5]
        if action_Scaling_list is None:
            action_Scaling_list = [1, 1, 1, 1, 1, 1]
        super().__init__(target_position=target_position, action_Scaling_list=action_Scaling_list)


def train(env, hyperparameters, actor_model, critic_model, device):
    """
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000).to(device=device)


def test(env, actor_model):
    """
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim).to(device=device)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)


#
# def localMain():
#     """
# 		The main function to run.
#
# 		Parameters:
# 			args - the arguments parsed from command line
#
# 		Return:
# 			None
# 	"""
#     # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
#     # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
#     # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
#     hyperparameters = {
#         'timesteps_per_batch': 2048,
#         # 'max_timesteps_per_episode': 200,
#         'max_timesteps_per_episode': 400,
#         'gamma': 0.99,
#         'n_updates_per_iteration': 10,
#         'lr': 3e-4,
#         'clip': 0.2,
#         'render': True,
#         'render_every_i': 10
#     }
#
#     # Creates the environment we'll be running. If you want to replace with your own
#     # custom environment, note that it must inherit Gym and have both continuous
#     # observation and action spaces.
#     # env = gym.make('Pendulum-v1')
#     # env = gym.make('Pendulum-v1')
#
#     import argparse
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # parser.add_argument('--seed', help='RNG seed', type=int, default=1)
#     parser.add_argument('--Target', help='Target Position', nargs='+', type=float, default=[-1, 0, 0.5])
#     parser.add_argument('--action_Scaling_list',
#                         help='action_Scaling_list', nargs='+', type=float, default=[1, 1, 1, 1, 1, 1])
#     parser.add_argument('--Name', help='Trail name', type=str, default="MPC_APR7")
#     # parser.add_argument('--H', help='MPC Looking ahead how many steps', type=int, default=5)
#     # parser.add_argument('--L', help='MPC second loop --- num of real actions', type=int, default=256)
#     parser.add_argument('--a_size', help='a_size', type=int, default=6)
#
#     parser.add_argument('--mode', help='Training mode', type=str, default="train")
#     parser.add_argument('--actor_model', help='actor_model', type=str, default="")
#     parser.add_argument('--critic_model', help='critic_model', type=str, default="")
#
#     args = parser.parse_args()
#
#     p.connect(p.GUI)
#     # p.connect(p.DIRECT)
#     # create a PyBullet env
#     print("target_position: ", args.Target)
#     env = Snake_PPO_V1(target_position=args.Target, action_Scaling_list=args.action_Scaling_list)
#     print("re-setting environment")
#     env.reset()
#
#     # Train or test, depending on the mode specified
#     if args.mode == 'train':
#         train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
#     else:
#         test(env=env, actor_model=args.actor_model)

def colab_Main(device):
    """
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--Target', help='Target Position', nargs='+', type=float, default=[1.5, 0, 1.5])
    parser.add_argument('--action_Scaling_list',
                        help='action_Scaling_list', nargs='+', type=float,
                        default=[1, 1, 1, 1, 1, 1])
                        # default=[1, 0, 1, 0, 1, 0])
                        # default=[1, 0, 0, 0, 1, 1])
    parser.add_argument('--Name', help='Trail name', type=str, default="MPC_APR7")
    # parser.add_argument('--H', help='MPC Looking ahead how many steps', type=int, default=5)
    # parser.add_argument('--L', help='MPC second loop --- num of real actions', type=int, default=256)
    parser.add_argument('--a_size', help='a_size', type=int, default=6)
    parser.add_argument('--max_timesteps_per_episode', help="max_timesteps_per_episode", type=int, default=10)

    parser.add_argument('--mode', help='Training mode', type=str, default="train")
    parser.add_argument('--actor_model', help='actor_model', type=str, default="")
    parser.add_argument('--critic_model', help='critic_model', type=str, default="")

    args = parser.parse_args()

    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        'timesteps_per_batch': 2048,
        # 'max_timesteps_per_episode': 200, # -74782858.53
        # 'max_timesteps_per_episode': 400,   # -150585164.9
        # 'max_timesteps_per_episode': 20,    # -7362647.79
        # 'max_timesteps_per_episode': 5,    # -1845846.55
        'max_timesteps_per_episode': args.max_timesteps_per_episode,  # -1107014.01
        'gamma': 0.99,
        # 'gamma': 0.9,
        'n_updates_per_iteration': 5,
        # 'n_updates_per_iteration': 10,
        # 'lr': 2.5e-4,
        'lr': 1e-2,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    # env = gym.make('Pendulum-v1')
    # env = gym.make('Pendulum-v1')

    # if device=="cpu":
    #     p.connect(p.GUI)
    # else:
    #     p.connect(p.DIRECT)

    # create a PyBullet env
    print("target_position: ", args.Target)
    env = Snake_PPO_V1(target_position=args.Target, action_Scaling_list=args.action_Scaling_list)
    print("re-setting environment")
    env.reset()

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,
              critic_model=args.critic_model, device=device)
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == "__main__":
    # set device to cpu or cuda
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
        p.connect(p.DIRECT)
        colab_Main(device)
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")
        p.connect(p.GUI)
        colab_Main(device)
