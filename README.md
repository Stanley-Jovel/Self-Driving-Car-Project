# Self-Driving Cars Faster in Godot Game Engine

This project explores leveraging behavior cloning (BC) to speed up Proximal Policy Optimization (PPO) training for self-driving cars in the Godot game engine.

- [View Results](https://stanley-jovel.github.io/Self-Driving-Car-Project/#results)

## Training Process
- Behavior cloning training: `train.ipynb`
- PPO training: `train_ppo.py`
- Evaluation: `policy.py`

## BC Pre-training
Trained a BC model to mimic expert driving behavior using `train.ipynb`.

## PPO Training
Trained PPO agent with and without BC pre-training using `train_ppo.py`, comparing their learning speeds and performances.

## Challenges
- Learning curve of Godot Engine
- Limited training data
- Complex architecture challenges

## Future Steps
- Develop a multi-agent environment
- Introduce obstacle avoidance
- Address multimodality using Behavior Transformer (BeT) models