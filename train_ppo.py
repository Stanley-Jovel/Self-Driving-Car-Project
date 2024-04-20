import time

import torch
import torch.nn as nn
from torch.optim import Adam
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.ppo_agent import PPOModel
from src.ppo_critic import PPOCritic
from src.agent import Constants

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters (adjust these based on your environment and needs)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5_000
MINI_BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_BETA = 0.01
NUM_STEPS_PER_EPOCH = 800
N_UPDATES_PER_ITERATION = 5

def ppo_update(actor, actor_optim, critic, critic_optim, obs, actions, log_probs, rewards, next_obs, dones, epoch):
    # Calculate advantages using Generalized Advantage Estimation (GAE)
    values = critic(obs).squeeze()
    # next_values = critic(next_obs)
    # advantages = calculate_gae(rewards, values, next_values, dones, GAMMA, GAE_LAMBDA)
    batch_rtgs = compute_rtgs(rewards, GAMMA)
    A_k = batch_rtgs - values.detach()
    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

    # PPO optimization iterations
    for _ in range(N_UPDATES_PER_ITERATION):
        mean_actor_loss = 0
        mean_critic_loss = 0
        
        dist = actor(obs)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - log_probs)
        # entropy = dist.entropy()

        # Calculate PPO loss
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        actor_loss = -torch.min(ratio * A_k, clipped_ratio * A_k).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        values = critic(obs).squeeze()
        critic_optim.zero_grad()
        critic_loss = nn.MSELoss()(values, batch_rtgs)
        critic_loss.backward()
        critic_optim.step()
        
    # mean_loss /= N_UPDATES_PER_ITERATION
    # writer.add_scalar("mean_loss/train", mean_loss, epoch)
    # with open("ppo_training_results.csv", "a") as f:
    #     f.write(f"{total_avg_loss},")
    

def compute_rtgs(rewards, gamma):
    batch_rtgs = []
    
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        batch_rtgs.insert(0, discounted_reward)

    return torch.tensor(batch_rtgs, dtype=torch.float32)

# Create environment
env = StableBaselinesGodotEnv(
    env_path=None, show_window=False, seed=0, n_parallel=1, speedup=1
)
env = VecMonitor(env)

# Create PPO agent and optimizer
agent = PPOModel(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, Constants.OUTPUT_SIZE.value).to(device)

agent.load_state_dict(torch.load("pretrained_model_dict_cuda.pt", map_location=device))
actor_optim = Adam(agent.parameters(), lr=LEARNING_RATE)

critic = PPOCritic(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, 1).to(device)
critic_optim = Adam(critic.parameters(), lr=0.005)

# Initialize history buffer
history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)

def update_history(obs):
    global history
    history = torch.cat([history[1:], obs])


start_time = time.time()

# Training loop
for epoch in range(1, NUM_EPOCHS+1):
    # Collect data for one epoch
    obs_list, reward_list, next_obs_list, done_list, log_prob_list = [], np.array([]), [], np.array([]), []
    history_tensor, next_history_tensor, actions_tensor = torch.empty(0), torch.empty(0), torch.empty(0)
    obs = env.reset()
    obs = torch.tensor(obs["obs"], dtype=torch.float32)
    update_history(obs)

    for step in range(NUM_STEPS_PER_EPOCH):
        # Sample action from policy
        with torch.no_grad():
            dist = agent(history.unsqueeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Interact with environment
        action = action.squeeze(0).detach().numpy()
        action = np.array([action])
        next_obs, reward, done, info = env.step(action)
        next_obs = torch.tensor(next_obs["obs"], dtype=torch.float32)

        # Store data
        history_tensor = torch.cat([history_tensor, history.unsqueeze(0)])
        actions_tensor = torch.cat([actions_tensor, torch.tensor(action)])
        reward_list = np.append(reward_list, reward)
        done_list = np.append(done_list, done)
        log_prob_list.append(log_prob)

        # Update observation and history
        update_history(obs)
        next_history_tensor = torch.cat([next_history_tensor, history.unsqueeze(0)])

    # Update PPO agent with collected data
    obs_tensor = history_tensor # torch.stack(history_tensor)
    rewards_tensor = torch.tensor(reward_list)
    next_obs_tensor = next_history_tensor # torch.stack(next_obs_list)
    dones_tensor = torch.tensor(done_list)
    log_probs_tensor = torch.stack(log_prob_list)

    ppo_update(agent, actor_optim, critic, critic_optim, obs_tensor, actions_tensor, log_probs_tensor, 
               rewards_tensor, next_obs_tensor, dones_tensor, epoch)
    
    mean_epoch_reward = np.mean(reward_list)
    
    print(f"Epoch {epoch} - Mean reward: {mean_epoch_reward}")
    # with open("ppo_training_results.csv", "a") as f:
        # f.write(f"{mean_epoch_reward}\n")
    # write mean_epoch_reward to tensorboard
    # writer.add_scalar("mean_epoch_reward/train", mean_epoch_reward, epoch)

    # Save trained agent
    torch.save(agent.state_dict(), "ppo_agent_trained_circle_track_clock_wise.pt")

print(f"--- {time.time() - start_time} seconds ---")

# Close tensorboard writer
writer.flush()
writer.close()
    
# Close environment
env.close()