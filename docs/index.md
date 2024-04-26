---
layout: default
---

- the training experiments
- the goal, never seen
- the architecture
- num history
- the training process
- BC vs no BC
- demo of the final result
- future steps
  obstacles, multi agents, multimodality

# Project Description

This project explores how leveraging behavior cloning (BC) can significantly reduce the training time required for a Proximal Policy Optimization (PPO) agent to master self-driving car maneuvers within the Godot game engine. By pre-training the agent with BC to mimic expert driving behavior, the PPO agent gains an initial understanding of the environment, leading to faster convergence and improved learning efficiency compared to training from scratch.

# BC Pre-training

The BC pre-training process involves training a neural network to predict the expert actions given the current state of the environment. The expert actions are obtained by recording the driving behavior of a human player. 

The BC pre-training process consists of the following steps:

1. **Data Collection**:

I drove the car manually in 4 different tracks in the game Godot environment to collect expert demonstrations. The expert demonstrations consist of state-action pairs, where the state represents the current observation of the environment, and the action represents the expert action taken by the human player.

Observations (States) include the following information:
- Car velocity X
- Car velocity Y
- Car angular velocity
- Car Steering Angle
- Raycast distances to obstacles in front of the car

Actions include the following:
- Acceleration
- Steering

The following image shows how the car "sees" the environment using raycasts:
<center><img width="600" alt="Car seeing the world through raycasts" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/bbfbc8e0-878b-43fa-8718-6665bb090257"></center>

The 4 tracks used for data collection are:
<table>
  <tr>
    <td>Track 1: Simple circle track with no obstacles</td>
    <td>Track 2: Track with turns</td>
    <td>Track 3: Track with sharper turns</td>
    <td>Track 4: 8 shaped track</td>
  </tr>
  <tr>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/51e7d152-0c7a-446c-bc0d-0e42509b1ee7"></td>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/dc05225c-ba4d-4ff1-be14-0723877638ce"></td>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/b389dddd-b6d5-4599-8081-06bfb88ef7f3"></td>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/d544a20b-4e0c-46cd-ba96-bc67e3d35f22"></td>
  </tr>
</table>

2. **Data Preprocessing**: Process the recorded data to extract state-action pairs.
3. **Neural Network Architecture**: Design a neural network architecture that takes the state as input and outputs the predicted action.
4. **Training**: Train the neural network using the state-action pairs obtained from the expert demonstrations.
5. **Evaluation**: Evaluate the performance of the trained BC model on unseen data to ensure it can accurately predict expert actions.




