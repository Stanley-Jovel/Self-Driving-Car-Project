---
layout: default
---

- [Home](#project-description)
- [Behavior Cloning (BC) Pre-training](#behavior-cloning-bc-pre-training)
- [Proximal Policy Optimization (PPO) Training](#proximal-policy-optimization-ppo-training)
- [Results](#results)
- [Challenges faced](#challenges-faced)
- [Future Steps](#future-steps)

You can find the code for this project [here]({{ site.github.repository_url }})

# Project Description

This project explores how leveraging behavior cloning (BC) can significantly reduce the training time required for a Proximal Policy Optimization (PPO) agent to master self-driving car maneuvers within the Godot game engine. By pre-training the agent with BC to mimic expert driving behavior, the PPO agent gains an initial understanding of the environment, leading to faster convergence and improved learning efficiency compared to training from scratch.

# Behavior Cloning (BC) Pre-training

The BC pre-training process involves training a neural network to predict the expert actions given the current state of the environment. The expert actions are obtained by recording the driving behavior of a human player. 

The BC pre-training process consists of the following steps:

1. **Data Collection**:

I drove the car manually in 4 different tracks in the Godot game environment to collect expert demonstrations. The expert demonstrations consist of state-action pairs, where the state represents the current observation of the environment, and the action represents the expert action taken by the human player.

Observations (States) are represented by 1D tensors with 25 elements, which include the following:
- Car velocity X
- Car velocity Y
- Car angular velocity
- Car Steering Angle
- 21 Raycast distances to obstacles in front of the car

Actions include the following:
- Acceleration
- Steering

The following image shows how the car "sees" the environment using raycasts:

![Figure 1](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/bbfbc8e0-878b-43fa-8718-6665bb090257)
*Figure 1: Car seeing the world through raycasts*

The 4 tracks used for data collection are:
<table>
  <tr>
    <td>#1: Circle-shaped track</td>
    <td>#2: Track with turns</td>
  </tr>
  <tr>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/51e7d152-0c7a-446c-bc0d-0e42509b1ee7"></td>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/dc05225c-ba4d-4ff1-be14-0723877638ce"></td>
  </tr>
  <tr>
    <td>#3: Track with sharper turns</td>
    <td>#4: Eight-shaped track</td>
  </tr>
  <tr>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/b389dddd-b6d5-4599-8081-06bfb88ef7f3"></td>
    <td><img width="600" src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/d544a20b-4e0c-46cd-ba96-bc67e3d35f22"></td>
  </tr>
</table>

![]()
*Figure 2: Tracks used for gathering expert data*

The idea is to collect enough expert demonstrations to cover a wide range of driving scenarios and behaviors, which will help the BC model generalize better to unseen data, and improve the performance of the PPO agent.

The final track we will use to evaluate how good the agent is at driving is the "Never Seen Track". This track was not used during the data collection phase, and the agent has never seen it before. This track is designed to test the generalization capabilities of the BC pre-trained agent:

![Never Seen](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/723ede0d-d965-411a-9fad-32ec415eac6e)
*Figure 3: Never seen track used to evaluate model*

2. **Neural Network Architecture**: 
I devised a neural network architecture that takes in historical observation states as input and outputs the predicted action. The idea behind using historical observations is to provide the model with temporal information about the environment, which can help it make better predictions. 

<center>

![Figure 4](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/5c1dfdbc-9921-4639-b556-d7d55903f44c)

*Figure 4: Passing historical observations to the neural network*
</center>

In order to process historical observations, I used a 1D convolutional neural network (CNN) architecture. The CNN architecture consists of the following layers:

```
- 1D Convolutional Layer (10 historic obs)
    ReLU Activation
    Max Pooling
    Batch Normalization
- 1D Convolutional
    ReLU Activation
    Max Pooling
    Batch Normalization
- Linear Layer
    ReLU Activation
- Linear Layer
    ReLU Activation
- Linear Layer
    ReLU Activation
- Linear Layer
    ReLU Activation
```

3. **Training**:
The neural network is trained using the collected expert demonstrations to minimize the distribution log loss between the predicted actions and the expert actions.

4. **Evaluation**:
The trained BC model is evaluated on the "Never Seen Track" to assess its performance in a new environment.

# Proximal Policy Optimization (PPO) Training

After pre-training the BC model, the PPO agent is trained using the BC model as an initialization. The PPO agent learns to improve its driving behavior through trial and error by interacting with the environment and receiving rewards based on its actions.

## Without BC Pre-training

To demonstrate the effectiveness of BC pre-training, I first trained the PPO agent from scratch without using any pre-trained model. The PPO agent learns to drive by interacting with the environment and receiving rewards based on its actions.

The following is a chart of the average rewards obtained by the PPO agent across 3 separate training runs in the "Never Seen Track":

![Figure 5](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/843f8647-aa8a-4f9e-9f7d-cc3096856922)
*Figure 5: Training progress of PPO without pretraining*

As shown in Figure 5, the PPO agent takes a significant number of episodes to learn how to drive effectively. The agent's performance is steady at the beginning, when driving in a straight line, but it struggles to navigate sharp turns. The agent's performance improves over time as it learns from its mistakes and explores different strategies.

## With BC Pre-training

Next, I trained the PPO agent using the BC pre-trained model as an initialization. The PPO agent starts with the knowledge of expert driving behavior, which helps it learn faster and achieve better performance. It does seems to struggle at the beginning, but it quickly learns to drive effectively.

The following is a comparison of the average rewards obtained by the PPO agent with and without BC pre-training across 3 separate training runs in the "Never Seen Track":

![Figure 6](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/d4ad27d0-4e0d-42b2-bebb-b5ab563bb664)
*Figure 6: Comparison of training progress with and without BC pretraining*

As shown in Figure 6, the PPO agent with BC pre-training achieves higher rewards and learns faster compared to the agent without pre-training. The agent's performance is more stable and consistent, and it learns to navigate sharp turns more effectively. Additionally, the agent takes much less time and less reward tunning to learn to drive fast.

## How long would it have taken to train without BC?

We can see that the pretrain model achieves decent reward returns as early as 300K learning steps, while the model without pretraining takes around 2M learning steps to achieve the same level of performance. This means that the BC pre-trained model is at least 85% more efficient in terms of learning speed.

![Figure 7](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/189266ff-bd55-43dd-803a-406b5124d493)
*Figure 7: Model comparison on 2M learning steps runs. Untrained model takes 2M steps to achieve the same performance as the pre-trained model at 300K steps*

# Results

<video controls>
  <source src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/0e16fb99-e2e7-495a-821c-4c9cd98b5a8b" type="video/mp4">
</video>

*Video 1: PPO agent driving in "Never Seen Track" without BC pre-training*

<br />
<br />

<video controls>
  <source src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/264e4eca-2866-44c0-ba87-8e2036f5028a" type="video/mp4">
</video>

*Video 2: PPO agent driving in "Never Seen Track" with BC pre-training*

<br />
<br />

# Challenges faced

I originally planned a much more ambitious project, I wanted to develop a symulation with multiple agents, obstacles and multimodalities. However, I faced several challenges that prevented me from achieving this goal:

1. **Godot Engine learning curve**: I had to learn how to use the Godot game engine from scratch, which took a significant amount of time. The lack of documentation and tutorials on using reinforcement learning in Godot made it challenging to implement complex algorithms.

2. **Training data**: I collected expert demonstrations of only 4 tracks, which may not be sufficient to cover all possible driving scenarios. More data would help the BC model generalize better and improve the performance of the PPO agent. However, that would have required more profiecient Godot skills to create more tracks.

3. **Complex architecture**: Contrary to what I expected, the complex architecture that worked well for BC did not work well for PPO. The PPO agent struggled to learn effectively with the BC pre-trained model due to the complexity of the neural network architecture. My experience suggests PPO agents work better with simpler architectures. A middle ground would have been ideal, where the BC model is simple enough for PPO but complex enough to capture the environment's dynamics. Time constraints prevented me from experimenting with this idea further.

4. **Historical observations**: Another counterintuitive finding was that the more historical observations I passed to the PPO agent, the worse its performance. This is because PPO already accounts for temporal information dealing with future rewards. I was effectively adding noise to the data, which hindered the agent's learning process.

# Future Steps

Despite the challenges faced, I believe this project has the potential to be further developed and improved. Some future steps include:

1. **Multi-agent environment**: Develop a simulation with multiple agents interacting with each other, which can help the agents learn more complex behaviors and strategies.

2. **Obstacle avoidance**: Introduce obstacles in the environment that the agent must avoid while driving, which can help the agent learn to navigate complex environments.

3. **Address multimodality**: Address multi-modality using Behavior Transformer (BeT) models. This'll allow the agent to properly handle scenarios where it is not clear if the agent should turn left or right.

Example:
![Figure 8](https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/9b4cef94-7500-4e82-85b0-e00f52d6ea46)

*Figure 8: Multi-modality in driving*

<video controls>
  <source src="https://github.com/Stanley-Jovel/Self-Driving-Car-Project/assets/1679438/93fbe30c-583d-4dd8-b265-e6fd287492fe" type="video/mp4">
</video>

*Video 3: agent facing multi-modality*


You can find the code for this project [here]({{ site.github.repository_url }})

