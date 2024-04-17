import torch
import torch.nn as nn
from torch.optim import Adam
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

from src.ppo_agent import PPOModel
from src.agent import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters (adjust these based on your environment and needs)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
MINI_BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_BETA = 0.01
NUM_STEPS_PER_EPOCH = 500

def ppo_update(model, optimizer, obs, actions, log_probs, rewards, next_obs, dones):
    # Calculate advantages using Generalized Advantage Estimation (GAE)
    _, values = model(obs)
    _, next_values = model(next_obs)
    advantages = calculate_gae(rewards, values, next_values, dones, GAMMA, GAE_LAMBDA)

    # Normalize advantages
    advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).unsqueeze(1)

    # PPO optimization epochs
    for _ in range(NUM_EPOCHS):
        # Shuffle data for mini-batch updates
        for idx in batch_sampler(MINI_BATCH_SIZE, obs.size(0)):
            # Get mini-batch data
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_log_probs = log_probs[idx]
            batch_advantages = advantages[idx].float()
            batch_returns = advantages[idx] + values[idx].detach()

            # Get new probabilities and values
            new_dist, new_values = model(batch_obs)
            new_log_probs = new_dist.log_prob(batch_actions)
            entropy = new_dist.entropy()

            # Calculate PPO loss
            ratio = torch.exp(new_log_probs - batch_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            critic_loss = nn.MSELoss()(new_values, batch_returns)
            actor_loss = actor_loss.float()
            critic_loss = critic_loss.float()
            total_loss = (actor_loss + critic_loss - ENTROPY_BETA * entropy.mean()).float()

            # Update model parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

def calculate_gae(rewards, values, next_values, dones, gamma, gae_lambda):
    # Generalized Advantage Estimation (GAE) calculation
    advantages = []
    gae = 0
    dones = dones.int()
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

def batch_sampler(batch_size, data_size):
    # Generates random mini-batches of indices
    for _ in range(data_size // batch_size):
        yield torch.randint(0, data_size, (batch_size,))

# Create environment
env = StableBaselinesGodotEnv(
    env_path=None, show_window=False, seed=0, n_parallel=1, speedup=1
)
env = VecMonitor(env)

# Create PPO agent and optimizer
agent = PPOModel(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value, Constants.OUTPUT_SIZE.value).to(device)
optimizer = Adam(agent.parameters(), lr=LEARNING_RATE)

# Initialize history buffer
history = torch.zeros(Constants.NUM_HISTORY.value, Constants.INPUT_SIZE.value)

def update_history(obs):
    global history
    history = torch.cat([history[1:], obs])

# Training loop
for epoch in range(NUM_EPOCHS):
    # Collect data for one epoch
    obs_list, reward_list, next_obs_list, done_list, log_prob_list = [], np.array([]), [], np.array([]), []
    history_tensor, next_history_tensor, actions_tensor = torch.empty(0), torch.empty(0), torch.empty(0)
    obs = env.reset()
    obs = torch.tensor(obs["obs"], dtype=torch.float32)
    update_history(obs)

    for step in range(NUM_STEPS_PER_EPOCH):
        # Sample action from policy
        with torch.no_grad():
            dist, _ = agent(history.unsqueeze(0))
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

    ppo_update(agent, optimizer, obs_tensor, actions_tensor, log_probs_tensor, 
               rewards_tensor, next_obs_tensor, dones_tensor)

# Close environment
env.close()