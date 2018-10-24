"""
Actor-critic model.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class LowDimActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(LowDimActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #print('state:  {}'.format(state.shape))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        #print('out:  {}'.format(x.shape))
        return x

class LowDimCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(LowDimCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size+action_size*2, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)  # TODO paramaterize by n_agents
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #print('state pre:  {}'.format(state.shape))
        #state = state.reshape((-1, 48))
        #print('action pre:  {}'.format(action.shape))
        #action = action.reshape((-1, 4))
        #print('action pst:  {}'.format(action.shape))
        #print('state pst:  {}'.format(state.shape))
        xs = torch.cat((state, action), dim=1)
        x = F.relu(self.fcs1(xs))

        #print('x:  {}'.format(x.shape))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print('pre:  {}'.format(x.shape))
        #x = x.reshape((-1, 2))
        #print('out:  {}'.format(x.shape))
        return x

# Initialize local and target network with identical initial weights.

class LowDim2x():
    def __init__(self, state_size=24, action_size=2, seed=0):
        self.actor_local = LowDimActor(state_size, action_size, seed).to(device)
        self.actor_target = LowDimActor(state_size, action_size, seed).to(device)
        self.critic_local = LowDimCritic(state_size*2, action_size, seed).to(device)  # TODO paramaterize by n_agents
        self.critic_target = LowDimCritic(state_size*2, action_size, seed).to(device)  # TODO paramaterize by n_agents
        print(self.actor_local)
        summary(self.actor_local, (state_size,))
        print(self.critic_local)
