import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, Independent


class ActionContinuous(nn.Module):
    def __init__(self, args, action_dim):
        super().__init__()
        self.args = args
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ELU(),
        )
        self.mu = nn.Linear(args.hidden_size, action_dim)
        self.std = nn.Linear(args.hidden_size, action_dim)

    def forward(self, state, deterministic, training=True):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        mu = self.mu(x)
        std = self.std(x)
        std = F.softplus(std + 5.) + self.args.min_std

        mu = mu / self.args.mean_scale
        mu = torch.tanh(mu)
        mu = mu * self.args.mean_scale

        action_dist = Normal(mu, std)
        action_dist = torch.distributions.TransformedDistribution(
            action_dist, torch.distributions.TanhTransform())
        action_dist = Independent(action_dist, 1)
        action = action_dist.rsample()
        return action_dist, action


class ActionDiscrete(nn.Module):
    def __init__(self, args, action_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        self.logits = nn.Linear(args.hidden_size, action_dim)

    def forward(self, state, deterministic, training=True):
        x = self.seq(torch.cat([state, deterministic], dim=-1))
        logits = self.logits(x)
        dist = OneHotCategorical(logits=logits)

        if training:
            action_raw = dist.sample()                          # valid for log_prob
            action = action_raw + dist.probs - dist.probs.detach()  # STE
            return dist, action, action_raw
        else:
            action_raw = F.one_hot(dist.probs.argmax(dim=-1), num_classes=logits.size(-1)).float()
            action = action_raw
            return dist, action
