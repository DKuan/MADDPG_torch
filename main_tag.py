# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import os

import time
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

from model import actor_agent, critic_agent
from memory import Memory
from replay_buffer import ReplayBuffer
from arguments import parse_args
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

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

def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist):
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    if arglist.restore == True: # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name+'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name+'t_{}'.format(agent_idx))

    for i in range(env.n):
        actors_cur[i] = actor_agent(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = critic_agent(input_size_global, arglist).to(arglist.device)
        actors_tar[i] = actor_agent(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = critic_agent(input_size_global, arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                    (1-tao)*state_dict_t[key] 
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    print('=============================')
    print('=1 Env is right ...')
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].n-1 for i in range(env.n)] # no need for stop bit
    num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)
    
    print('=2 The agents are inited ...')
    print('=============================')

    """step3: init the pars """
    var = arglist.var
    var_discount = arglist.var_discount 
    collision_cnt = 0
    train_step = 0
    episode_step = 0
    obs_n = env.reset()
    agent_update_cnt = 0
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)] # set the init reward
    final_ep_rewards = [] # sum of rewards for training curve
    final_ep_ag_rewards = [] # agent rewards for training curve
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward
    head = 0
    end = 0
    obs_size = []
    for obs_shape in obs_shape_n:
        end = end + obs_shape
        range_obs = (head, end)
        obs_size.append(range_obs)
        head = end

    print('=3 starting iterations ...')
    print('=============================')

    for train_step in range(arglist.max_steps):
        if train_step % 100 == 0:   print('                                   var:{} \
         The ave200 reward is {}'.format(round(var, 4), round(np.mean(episode_rewards[-200:-1]), 4)), end='\r')
        print('=Training: the episode is {}'.format(train_step), end='\r')

        # get action
        out = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
            for agent, obs in zip(actors_cur, obs_n)]
        out_noise = [np.clip(np.random.normal(a, var), -1, 1) for a in out]
        action_n = [np.concatenate([np.array([0]), a_n])  for a_n in out_noise]

        # interact with env
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        if arglist.scenario_name == 'simple_adversary':
            rew_n_true = [r_n - r_n_o for r_n, r_n_o in zip(rew_n, rew_n_old)]
            if any([rew < -70 for rew in rew_n]): done = True
        if arglist.scenario_name == 'simple_tag':
            rew_n_true = rew_n
            if any([rew < -10 for rew in rew_n]): done = True
            if any([rew == 10 for rew in rew_n[:-1]]): collision_cnt += 1
        if arglist.scenario_name == 'simple_spread':
            rew_n_true = rew_n
            if any([rew < -25 for rew in rew_n]): done = True
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # save the experience
        memory.add(obs_n, np.concatenate(out_noise), rew_n_true , new_obs_n, done_n)

        # update the obs_n
        obs_n, rew_n_old, episode_step = new_obs_n, rew_n, episode_step+1
        
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # check if the game is end
        if done or terminal:
            episode_step = 0
            obs_n = env.reset()
            rew_n_old = [0.0 for _ in range(env.n)]
            agent_info.append([[]])
            episode_rewards.append(0)
            for a_r in agent_rewards:   a_r.append(0)
            
        # update all trainers, if not in display or benchmark mode
        if train_step > arglist.learning_start_step:
            if var > 0.08: var *= var_discount
            if (train_step - arglist.learning_start_step) % arglist.learning_fre == 0:
                agent_update_cnt += 1
                # update the target par using the cur
                actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao) 
                critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao) 

                for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
                    if opt_c == None: continue # jump to the next model update
                    loss_c_g = torch.tensor(0.0, device=arglist.device, requires_grad=True)
                    loss_a_g = torch.tensor(0.0, device=arglist.device, requires_grad=True)
                    # sample the experience
                    _obs_n, _action_n, _rew_n, _obs_n_new, _done_n = memory.sample( \
                        arglist.batch_size, agent_idx)
                    # use the data to update CRITIC 
                    for obs_n_o, action_cur, rew, obs_n_n, done in \
                            zip(_obs_n, _action_n, _rew_n, _obs_n_new, _done_n):
                        # critic loss:get the critic input
                        rew = torch.tensor(rew, device=arglist.device) # set the rew to gpu
                        done = torch.tensor(~done, dtype=torch.float, device=arglist.device) # set the rew to gpu
                        action_cur = torch.from_numpy(action_cur).to(arglist.device, torch.float)
                        obs_n_o = torch.from_numpy(obs_n_o).to(arglist.device, torch.float)
                        obs_n_n = torch.from_numpy(obs_n_n).to(arglist.device, torch.float)
                        action_tar = torch.cat([a_t(obs_n_n[obs_size[idx][0]:obs_size[idx][1]]).detach() \
                            for idx, a_t in enumerate(actors_tar)])
                        input_g_o = torch.cat([obs_n_o, action_cur])
                        input_g_n = torch.cat([obs_n_n, action_tar])
                        q = critic_c(input_g_o)
                        q_ = critic_t(input_g_n)
                        # critic loss:cal the loss
                        loss_c = torch.pow(q_*arglist.gamma*done + rew - q, 2) # bellman equation
                        loss_c = torch.div(loss_c, arglist.batch_size)  
                        #loss_c.backward(retain_graph=True)
                        loss_c_g = torch.add(loss_c_g, loss_c)
                        #loss_c_g = torch.add(loss_c_g, loss_c) if loss_c_g != None else loss_c
                    opt_c.zero_grad()
                    loss_c_g.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                    opt_c.step()

                    # use the data to update the ACTOR
                    for obs_n_o, action_cur, rew, obs_n_n, done in \
                            zip(_obs_n, _action_n, _rew_n, _obs_n_new, _done_n):
                        rew = torch.tensor(rew, device=arglist.device) # set the rew to gpu
                        obs_n_o = torch.from_numpy(obs_n_o).to(arglist.device, torch.float)
                        #obs_old = obs_n_o[obs_size[agent_idx][0]:obs_size[agent_idx][1]]
                        policy_actor = torch.cat([a_c(obs_n_o[obs_size[idx][0]:obs_size[idx][1]]) \
                           for idx, a_c in enumerate(actors_cur)])
                        #input_g = torch.cat([obs_old, action_cur])
                        input_g = torch.cat([obs_n_o, policy_actor])
                        loss_a = torch.mul(-1, critic_c(input_g))
                        loss_a = torch.div(loss_a, arglist.batch_size)
                        loss_a_g = torch.add(loss_a_g, loss_a)
                    opt_a.zero_grad()
                    loss_a_g.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                    opt_a.step()
                
                # save the model to the path_dir ---cnt by update number
                if agent_update_cnt > arglist.start_save_model and agent_update_cnt % arglist.episode4save_model == 0:
                    time_now = time.strftime('%y%m_%d%H%M')
                    print('=time:{} step:{},var:{}  save'.format(time_now, train_step, var))
                    file_txt = open(os.path.join('logs', 'record_{}.txt'.format(arglist.start_time)), 'a')
                    file_txt.writelines('save model time:{} step:{}, var:{} collision_cnt {}\n'.format(\
                        time_now, train_step, var, collision_cnt))
                    file_txt.close()
                    model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                        arglist.scenario_name, time_now, train_step))
                    if not os.path.exists(model_file_dir):
                        os.mkdir(model_file_dir)
                        file_txt = open(os.path.join(model_file_dir, 'readme.txt'), 'a')
                        file_txt.writelines('obs 1+2+3 +5*3 for critic\n')
                        file_txt.writelines('backward -1\n')
                        file_txt.writelines('relu [-1 1] limit, early stop\n')
                        file_txt.writelines('nn {} {}'.format(arglist.num_units_1, arglist.num_units_2))
                        file_txt.close()
                    for agent_idx, (a_c, a_t, c_c, c_t) in \
                        enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                        torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                        torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                        torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                        torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
