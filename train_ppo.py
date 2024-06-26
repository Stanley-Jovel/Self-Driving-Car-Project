import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.ppo_agent import PPOModel, Constants
from src.ppo_critic import PPOCritic

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--gamma",
    default=0.99,
    type=float,
    help="Discount factor for rewards",
)
parser.add_argument(
    "--clip",
    default=0.2,
    type=float,
    help="Clip Epsilon for PPO",
)
parser.add_argument(
    "--entropy",
    default=0.01,
    type=float,
    help="Entropy Beta for PPO",
)
parser.add_argument(
    "--env_path",
    default=None,
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument(
    "--pretrained_bc",
    default=False,
    action="store_true",
    help="Use pretrained behavior cloning model",
)
args, _ = parser.parse_known_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (adjust these based on your environment and needs)
LEARNING_RATE = 1e-3
TOTAL_TIMESTEPS = 500_000
TIMESTEPS_PER_EPISODE = 4000
TIMESTEPS_PER_BATCH = TIMESTEPS_PER_EPISODE * 3
N_UPDATES_PER_ITERATION = 3

writer = SummaryWriter(log_dir=f"runs/pretrained_{args.pretrained_bc}_gamma_{args.gamma}_clip_{args.clip}_entropy_{args.entropy}_updates_per_iteration_{N_UPDATES_PER_ITERATION}")

# Create PPO agent and optimizer
agent = PPOModel(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, Constants.OUTPUT_SIZE.value).to(device)
if args.pretrained_bc:
    agent.load_state_dict(torch.load("pretrained_model_dict_cuda_num_history_1.pt", map_location=device))
# agent.load_state_dict(torch.load("ppo_agent_trained_circle_track_clock_wise.pt", map_location=device))
actor_optim = Adam(agent.parameters(), lr=LEARNING_RATE)

critic = PPOCritic(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, 1).to(device)
# critic.load_state_dict(torch.load("ppo_critic_trained_circle_track_clock_wise.pt", map_location=device))
critic_optim = Adam(critic.parameters(), lr=LEARNING_RATE)

actor_sched = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=2, gamma=0.9)
critic_sched = torch.optim.lr_scheduler.StepLR(critic_optim, step_size=2, gamma=0.9)

def ppo_update(obs, actions, log_probs, rewards, dones, t_so_far):
    # Calculate advantages using Generalized Advantage Estimation (GAE)
    agent.train()
    critic.train()

    values = critic(obs).squeeze()
    batch_rtgs = compute_rtgs(rewards, args.gamma)
    A_k = batch_rtgs - values.detach()
    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

    mean_actor_loss = 0
    mean_critic_loss = 0
    # PPO optimization iterations
    for _ in range(N_UPDATES_PER_ITERATION):
        dist = agent(obs)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - log_probs)

        # Calculate PPO loss
        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
        actor_loss = -torch.min(ratio * A_k, clipped_ratio * A_k).mean()
        actor_loss -= args.entropy * entropy
        mean_actor_loss += actor_loss.item()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        values = critic(obs).squeeze()
        critic_optim.zero_grad()
        critic_loss = nn.MSELoss()(values, batch_rtgs)
        mean_critic_loss += critic_loss.item()
        critic_loss.backward()
        critic_optim.step()

    mean_actor_loss /= N_UPDATES_PER_ITERATION
    mean_critic_loss /= N_UPDATES_PER_ITERATION

    writer.add_scalar("mean_actor_loss/train", mean_actor_loss, t_so_far)
    writer.add_scalar("mean_critic_loss/train", mean_critic_loss, t_so_far)
    mean_reward = rewards.mean()
    writer.add_scalar("mean_reward/train", mean_reward, t_so_far)
    print(f"T So Far {t_so_far}, Mean reward: {mean_reward:.4f}, actor loss: {mean_actor_loss:.4f}, critic loss: {mean_critic_loss:.2f}")

def compute_rtgs(rewards, gamma):
    batch_rtgs = []
    
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        batch_rtgs.insert(0, discounted_reward)

    return torch.tensor(batch_rtgs, dtype=torch.float32)

# Create environment
env = StableBaselinesGodotEnv(
    env_path=args.env_path, show_window=False, seed=0, n_parallel=1, speedup=15
)
env = VecMonitor(env)

# Initialize history buffer
history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)

def update_history(obs):
    global history
    history = torch.cat([history[1:], obs])

start_time = time.time()

# Training loop
batch_obs = torch.empty(0, dtype=torch.float32)
batch_rewards = []
batch_log_probs = []
t_so_far = 0
while t_so_far < TOTAL_TIMESTEPS:
    # Collect data for one epoch
    history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)
    reward_list, done_list, log_prob_list = np.array([]), np.array([]), []
    batch_history, actions_tensor = torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    
    t = 0
    while t < TIMESTEPS_PER_BATCH:
        obs = env.reset()
        history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)
        obs = torch.tensor(obs["obs"], dtype=torch.float32)
        update_history(obs)

        for ep_t in range(TIMESTEPS_PER_EPISODE):
            # Sample action from policy
            agent.eval()
            with torch.no_grad():
                dist = agent(history.unsqueeze(0))
                action = dist.sample()
                log_prob = dist.log_prob(action)

            t += 1

            # Interact with environment
            action = action.squeeze(0).detach().numpy()
            action = np.array([action])
            actions_tensor = torch.cat([actions_tensor, torch.tensor(action, dtype=torch.float32)])
            next_obs, reward, done, info = env.step(action)

            # Store data
            reward_list = np.append(reward_list, reward)
            done_list = np.append(done_list, done)
            log_prob_list.append(log_prob)
            # Update observation and history

            batch_history = torch.cat([batch_history, history.unsqueeze(0)])
            next_obs = torch.tensor(next_obs["obs"], dtype=torch.float32)
            update_history(next_obs)
        
        t_so_far += ep_t + 1

        # Update PPO agent with collected data
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32)
        done_tensor = torch.tensor(done_list, dtype=torch.float32)
        log_prob_tensor = torch.tensor(log_prob_list, dtype=torch.float32).flatten()

    ppo_update(batch_history, actions_tensor, log_prob_tensor, reward_tensor, done_tensor, t_so_far)
    
    # Save trained agent
    torch.save(agent.state_dict(), "ppo_agent_trained_circle_track_clock_wise.pt")
    torch.save(critic.state_dict(), "ppo_critic_trained_circle_track_clock_wise.pt")

print(f"--- {time.time() - start_time} seconds ---")

# Close tensorboard writer
writer.flush()
writer.close()
    
# Close environment
env.close()