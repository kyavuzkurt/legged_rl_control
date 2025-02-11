# Legged RL Control (WIP)
ROS 2 package for reinforcement learning-based control of legged robots using MuJoCo.

This package is a work in progress and is not ready for production use. 

## Supported Robots
- Unitree A1

## Installation

The  [a1_description package](https://github.com/kyavuzkurt/a1_description.git) is required to run the simulation.

```bash
cd ~/ros2_ws/src
git clone https://github.com/kyavuzkurt/legged_rl_control.git && git clone https://github.com/kyavuzkurt/a1_description.git
cd ..
colcon build --symlink-install
source install/setup.bash
```

## Usage

You can launch the simulation with the following command:

```bash
ros2 launch legged_rl_control a1_sim.launch.py
```

And you can train a policy with the following command on the workspace:

```bash
# Training
python scripts/train_a1.py

# Evaluation
python scripts/train_a1.py --eval --model-path path/to/ppo_model.zip

# Continue training
python scripts/train_a1.py --continue-training path/to/ppo_checkpoint.zip
```


## TODO
- Add ~~observation space normalization~~ and domain ~~randomization~~
- Implement curriculum learning and safety monitoring
- Improve reward function and policy deployment
- Add visualization tools and hardware interface
- Implement benchmarking and failure recovery
- Add motion primitives and parameter configuration
