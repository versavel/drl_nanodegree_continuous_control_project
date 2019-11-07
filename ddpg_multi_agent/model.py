import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=[256, 128], seed=0):
        """Initialize parameters and build model.
        
        Arguments:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size [int] : list with number of nodes in each hidden layer
            seed (int): seed for random number generator
        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=0.01)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01)
        #self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        #self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=[256, 128, 128], seed=0):
        """Initialize parameters and build model.

        Arguments:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size [int] : list with number of nodes in each hidden layer
            seed (int): seed for random number generator
        """
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0]+action_size, hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fcs1.weight.data, a=0.01)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01)
        torch.nn.init.kaiming_normal_(self.fc3.weight.data, a=0.01)
        #self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        #self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
