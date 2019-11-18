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

        self.LRuLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.linear_a1(input)
        x = self.LRuLU(x)
        #x = self.tanh(x)
        x = self.linear_a2(x)
        x = self.LRuLU(x)
        #x = self.tanh(x)
        policy = self.tanh(self.linear_a(x))
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, num_inputs_g, args):
        super(critic_agent, self).__init__()
        self.linear_c1 = nn.Linear(num_inputs_g, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LRuLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        self.linear_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, input_g):
        """
        input_g: input_global, input features of all agents
        """
        x = self.linear_c1(input_g)
        x = self.LRuLU(x)
        x = self.linear_c2(x)
        x = self.LRuLU(x)
        value = self.linear_c(x)
        return value
