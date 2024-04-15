import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, input_size, obs_list, action_list, num_history):
        self.input_size = input_size
        self.obs_list = obs_list
        self.action_list = action_list
        self.num_history = num_history

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs = self.obs_list[idx]
        action = self.action_list[idx]

        # Retrieve history observations
        start_idx = max(0, idx - self.num_history)
        history_obs = self.obs_list[start_idx:idx]

        # Pad history observations if necessary
        if len(history_obs) < self.num_history:
            pad_width = self.num_history - len(history_obs)
            padding = torch.zeros(pad_width, self.input_size)
            history_obs = torch.cat([padding, history_obs])

        return history_obs, action