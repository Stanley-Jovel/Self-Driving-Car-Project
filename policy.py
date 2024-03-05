import argparse
from train import RNN, Constants

import numpy as np
import torch
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

args, extras = parser.parse_known_args()

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

# load model from file
state_dict = torch.load("./model.pt")
model = RNN(
    Constants.INPUT_SIZE.value, 
    Constants.HIDDEN_SIZE.value, 
    Constants.NUM_LAYERS.value, 
    Constants.OUTPUT_SIZE.value, 
    Constants.DROPOUT.value)
model.load_state_dict(state_dict)
model.eval()

# inference
obs = env.reset()
obs = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
for _ in range(args.timesteps):
    with torch.no_grad():
        action = model(obs)
    action = action.squeeze(0).detach().numpy()
    action = np.array([action])
    print('action: ', action)
    obs, reward, done, info = env.step(action)
    obs = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)

close_env()
