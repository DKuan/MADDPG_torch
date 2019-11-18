# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch

import time
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")
    # environment
    #parser.add_argument("--scenario_name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--scenario_name", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--max-episode-len", type=int, default=75, help="maximum episode length")
    parser.add_argument("--max_steps", type=int, default=850000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # core training parameters
    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--learning_start_step", type=int, default=4e4, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=2e2, help="learning frequency")
    parser.add_argument("--var", type=int, default=1.2, help="var of the noise")
    parser.add_argument("--var_discount", type=float, default=0.999996, help="the discount for var")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=7e-3, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=700, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=3e5, help="number of data stored in the memory")
    parser.add_argument("--num_units_1", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=128, help="number of units in the mlp")
    # checkpointing
    parser.add_argument("--exp-name", type=str, default="maddpg", help="name of the experiment")
    parser.add_argument("--episode4save_model", type=int, default=70, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=1050, help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="models", help="directory in which training state and model should be saved")
    parser.add_argument("--old_model_name", type=str, default="models/1911_122134_20000/", help="directory in which training state and model are loaded")
    parser.add_argument("--restore_idxs", type=list, default=[0], help="the idx of agents need to restore from the model trained")
    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
