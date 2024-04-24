import argparse

import numpy as np
import torch
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

from src.agent import BehaviorCloningModel, Constants
from src.ppo_agent import PPOModel

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
    "--load_from_device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
)
device =  torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('device: ', device)
args, extras = parser.parse_known_args()
model = None

# Multi agent. try transformers, bet, gaussian mixture models (GMM), diffussion. BC + RL. reward shaping. reverse RL.
# AI to AI, have an RL policy to learn good performance, run demos on it. check if rl demos are better at bc
   
# Instantiate model, loss function, and optimizer
# model = BehaviorCloningModel(
#     Constants.NUM_HISTORY.value, 
#     Constants.INPUT_SIZE.value, 
#     Constants.OUTPUT_SIZE.value).to(device)

model = PPOModel(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, Constants.OUTPUT_SIZE.value).to(device)
model.load_state_dict(torch.load(f"pretrained_model_dict_{args.load_from_device}.pt", map_location=device))
# model.load_state_dict(torch.load(f"ppo_agent_trained_circle_track_clock_wise.pt", map_location=device))
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
        # action = model(history.unsqueeze(0))
        dist = model(history.unsqueeze(0))
        action = dist.sample()
    action = action.squeeze(0).detach().numpy()
    action = np.array([action])
    obs, reward, done, info = env.step(action)
    obs = torch.tensor(obs["obs"], dtype=torch.float32)
    update_history(obs)

close_env()
