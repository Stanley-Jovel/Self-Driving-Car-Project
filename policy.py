import argparse
import os
import json
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--timesteps",
    default=1_000_000,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000. If resuming from a saved model, "
    "it will continue training for this amount of steps from the saved state without counting previously trained "
    "steps",
)
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
)

args, extras = parser.parse_known_args()
model = None

    # Define RNN model
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(FNN, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.policy(x)
        return x
    
class Constants(Enum):
    INPUT_SIZE = 25  # Number of features in observation
    HIDDEN_SIZE = 128  # Number of units in hidden layer
    NUM_HISTORY = 3  # Number of history steps to use
    OUTPUT_SIZE = 2  # Number of actions
    DROPOUT = 0.5  # Dropout rate
    lr = 1e-3  # Learning rate
    EPOCHS = 20  # Number of epochs to train

    NUM_LAYERS = 4  # Number of LSTM layers

def train():
    global model
    obs_list = torch.tensor([])
    action_list = torch.tensor([])

    for file in os.listdir("./demos"):
        if file.startswith("*"):
            continue
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
    class Dataset(torch.utils.data.Dataset):
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

    dataset = Dataset(obs_list, action_list)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    iterator = tqdm(range(Constants.EPOCHS.value), total=Constants.EPOCHS.value, desc="Training")

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

if args.train:
    train()
else:
    model = torch.load("./model.pt")

model.eval()
def close_env():
    try:
        print("closing env")
        env.close()
    except Exception as e:
        print("Exception while closing env: ", e)

env = StableBaselinesGodotEnv(
    env_path=None, show_window=False, seed=0, n_parallel=1, speedup=1
)
env = VecMonitor(env)

history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)

def update_history(obs):
    global history
    history = torch.cat([history[1:], obs])

# inference
obs = env.reset()
obs = torch.tensor(obs["obs"], dtype=torch.float32)
update_history(obs)
for _ in range(args.timesteps):
    with torch.no_grad():
        obs = torch.cat([o for o in history])
        action = model(obs)
    action = action.squeeze(0).detach().numpy()
    action = np.array([action])
    obs, reward, done, info = env.step(action)
    obs = torch.tensor(obs["obs"], dtype=torch.float32)
    update_history(obs)

close_env()
