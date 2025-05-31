import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ComposeTransform, TanhTransform

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_layers, action_dim):
        super(PolicyNet, self).__init__()
        self.dist_transform = ComposeTransform([
            TanhTransform(cache_size=1),
        ])

        self.flatten = nn.Flatten()
        self.backbone = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        ])
        for _ in range(num_layers - 2):
            self.backbone.append(nn.Linear(hidden_dim, hidden_dim))
            self.backbone.append(nn.ReLU())
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs):
        x = self.flatten(obs)
        for module in self.backbone:
            x = module(x)
        mean = self.mean_head(x)
        std = self.log_std_head(x)
        return mean, std
    
    def sample(self, obs):
        mean, log_std = self(obs)
        std = torch.clamp(log_std, min=-20., max=20.).exp()
        normal = TransformedDistribution(Normal(mean, std), self.dist_transform)
        samples = normal.rsample()
        log_prob = normal.log_prob(samples).sum(-1, keepdim=True)
        return samples, log_prob
    
class SoftQNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_layers, action_dim):
        super(SoftQNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
        ])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 1))
    
    def forward(self, obs, action):
        obs = self.flatten(obs)
        action = self.flatten(action)
        x = torch.concat([obs, action], dim=-1)
        for module in self.layers:
            x = module(x)
        return x
    
class SoftVNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_layers):
        super(SoftVNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        ])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 1))
    
    def forward(self, obs):
        x = self.flatten(obs)
        for module in self.layers:
            x = module(x)
        return x
