import argparse

import numpy as np
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


obs = env.reset()
for action in actions:
    action = np.array([action])
    # action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

close_env()
