# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        # policy = F.softmax(policy) # use softmax to get the policy
        # action_out = policy.multinomial(1) # get the max action
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        #x = self.tanh(x)
        x = self.LReLU(self.linear_a2(x))
        #x = self.tanh(x)
        policy = self.tanh(self.linear_a(x))
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1*2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_openai)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai*2, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        #x = self.tanh(x)
        x = self.LReLU(self.linear_a2(x))
        #x = self.tanh(x)
        policy = self.linear_a(x)
        u = torch.rand_like(policy)
        policy = F.softmax(policy - torch.log(-torch.log(u)), dim=0) 
        return policy
