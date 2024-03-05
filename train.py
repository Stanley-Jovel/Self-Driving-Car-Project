import os
import json
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

class Constants(Enum):
    INPUT_SIZE = 25  # Number of features in observation
    HIDDEN_SIZE = 64  # Number of units in hidden layer of RNN
    OUTPUT_SIZE = 2  # Number of actions
    NUM_LAYERS = 4  # Number of LSTM layers
    DROPOUT = 0.5  # Dropout rate
    lr = 1e-3  # Learning rate

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
class Dataset(Dataset):
    def __init__(self, obs_list, action_list):
        self.obs_list = obs_list
        self.action_list = action_list

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        return self.obs_list[idx], self.action_list[idx]
    

dataset = Dataset(obs_list, action_list)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use only the last output in the sequence
        return out

# Instantiate model, loss function, and optimizer
model = RNN(
    Constants.INPUT_SIZE.value, 
    Constants.HIDDEN_SIZE.value,
    Constants.NUM_LAYERS.value, 
    Constants.OUTPUT_SIZE.value,
    Constants.DROPOUT.value)

# create a loss function
loss_fn = nn.MSELoss()

# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=Constants.lr.value)

# train the model
epochs = 40
iterator = tqdm(range(epochs), total=epochs, desc="Training")

for epoch in iterator:
    model.train()
    for obs, action in train_dataloader:
        optimizer.zero_grad()
        pred = model(obs.unsqueeze(1))
        loss = loss_fn(pred, action)
        loss.backward()
        optimizer.step()
    iterator.set_postfix(epoch=epoch, loss=loss.item())

# evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for obs, action in test_dataloader:
        pred = model(obs.unsqueeze(1))
        test_loss += loss_fn(pred, action).item()
    test_loss /= len(test_dataloader)
    print(f"Test loss: {loss.item()}")

# save the model
torch.save(model.state_dict(), "model.pt")