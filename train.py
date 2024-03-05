import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
# train data are json files located in ./demos folder.
# each item in the json file is an episode compose of an array of 2 elements: the observation and the action.

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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# create a model to train on obs_list and action_list
model = nn.Sequential(
    nn.Linear(21, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Tanh()
)

# create a loss function
loss_fn = nn.MSELoss()

# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train the model
epochs = 10
iterator = tqdm(range(epochs), total=epochs, desc="Training")

for epoch in iterator:
    for obs, action in train_dataloader:
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iterator.set_postfix(epoch=epoch, loss=loss.item())

# evaluate the model
with torch.no_grad():
    for obs, action in test_dataloader:
        pred = model(obs)
        loss = loss_fn(pred, action)
    print(f"test loss: {loss.item()}")

# save the model
torch.save(model, "model.pt")