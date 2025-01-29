import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.actor = PolicyNetwork(obs_size, act_size)
        self.critic = PolicyNetwork(obs_size, 1)
        
    def forward(self, x):
        return self.actor(x), self.critic(x) 