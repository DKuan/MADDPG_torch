import os
import sys

import torch
import torch.nn.functional as F
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

from model import actor_agent, critic_agent
from arguments import parse_args

def make_env(scenario_name, arglist, benchmark=False):
    """ 
    create the environment from script 
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tar = [torch.load(arglist.old_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(env.n)]

    return actors_tar

def enjoy(arglist):
    """ 
    This func is used for testing the model
    """

    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    """ init the agents """
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    actors_tar = get_trainers(env, arglist)

    """ interact with the env """
    obs_n = env.reset()
    while(1):

        # update the episode step number
        episode_step += 1

        # get action
        action_n = []
        for actor, obs in zip(actors_tar, obs_n):
            model_out,_ = actor(torch.from_numpy(obs).to(arglist.device, torch.float),model_original_out=True)
            action_n.append(F.softmax(model_out,dim=-1).detach().cpu().numpy())

        # interact with env
        obs_n, rew_n, done_n, info_n = env.step(action_n)

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset()

        # render the env
        #print(rew_n)
        #env.render()

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
