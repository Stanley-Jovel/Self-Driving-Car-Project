import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class PPOCritic(nn.Module):
    def __init__(self, num_history, num_features, output_size, action_std_init=0.6):
        super(PPOCritic, self).__init__()

        self.policy = nn.Sequential(
            nn.Conv1d(num_history, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64)
        )
        self.flattened_size = 64 * (num_features // 4)

        self.critic = nn.Sequential(
            nn.Linear(self.flattened_size, self.flattened_size),
            nn.ReLU(),
            nn.Linear(self.flattened_size, self.flattened_size),
            nn.ReLU(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.policy(x)
        x = x.view(-1, self.flattened_size)
        value = self.critic(x)
        return value