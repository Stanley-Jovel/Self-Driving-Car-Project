import os
import json
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class Constants(Enum):
    INPUT_SIZE = 25  # Number of features in observation
    HIDDEN_SIZE = 128  # Number of units in hidden layer of RNN
    OUTPUT_SIZE = 2  # Number of actions
    DROPOUT = 0.5  # Dropout rate
    lr = 1e-3  # Learning rate
    NUM_HISTORY = 25  # Number of history steps to use

    NUM_LAYERS = 4  # Number of LSTM layers

# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = None

def train():
    global model
    obs_list = torch.tensor([])
    action_list = torch.tensor([])

    for file in os.listdir("./demos"):
        with open(f"./demos/{file}", "r") as f:
            data = json.load(f)
            for episode in data:
                min_length = min(len(episode[0]), len(episode[1]))
                obs = episode[0][:min_length]
                action = episode[1][:min_length]

                if len(obs) == 0 or len(action) == 0:
                    continue

                obs = torch.tensor(obs, dtype=torch.float32)
                action = torch.tensor(action, dtype=torch.float32)
                obs_list = torch.cat([obs_list, obs])
                action_list = torch.cat([action_list, action])

    # create a dataset
    class DatasetHistoric(torch.utils.data.Dataset):
        def __init__(self, obs_list, action_list, num_history=Constants.NUM_HISTORY.value):
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
                history_obs = torch.cat([torch.zeros(pad_width, Constants.INPUT_SIZE.value), history_obs], dim=0)

            return torch.cat([o for o in history_obs]), action
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, obs_list, action_list):
            self.obs_list = obs_list
            self.action_list = action_list
            
        def __len__(self):
            return len(self.obs_list)

        def __getitem__(self, idx):
            obs = self.obs_list[idx]
            action = self.action_list[idx]

            return obs, action

    dataset = Dataset(obs_list, action_list)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define RNN model
    class FNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout):
            super(FNN, self).__init__()
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.policy(x)
            return x

    # Instantiate model, loss function, and optimizer
    model = FNN(
        Constants.INPUT_SIZE.value * Constants.NUM_HISTORY.value, 
        Constants.HIDDEN_SIZE.value, 
        Constants.OUTPUT_SIZE.value, 
        Constants.DROPOUT.value)

    # create a loss function
    loss_fn = nn.MSELoss()

    # create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.lr.value)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    # train the model
    epochs = 20
    iterator = tqdm(range(epochs), total=epochs, desc="Training")

    for epoch in iterator:
        model.train()
        for obs, action in train_dataloader:
            optimizer.zero_grad()
            pred = model(obs)
            loss = loss_fn(pred, action)
            loss.backward()
            optimizer.step()
        iterator.set_postfix(epoch=epoch, loss=loss.item())
        # scheduler.step()

    # evaluate the model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for obs, action in test_dataloader:
            pred = model(obs)
            test_loss += loss_fn(pred, action).item()
        test_loss /= len(test_dataloader)
        print(f"Test loss: {loss.item()}")

    # save the model
    torch.save(model, "model.pt")