import rclpy
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import torch
import argparse
from typing import Callable

# Configuration Management
def load_configs(pkg_name='legged_rl_control'):
    """Load configurations from YAML files"""
    pkg_path = get_package_share_directory(pkg_name)
    
    config_paths = {
        'robot': Path(pkg_path) / 'config/robots/a1_config.yaml',
        'training': Path(pkg_path) / 'config/training/a1_training.yaml',
        'controller': Path(pkg_path) / 'config/controllers/pid.yaml'
    }
    
    configs = {}
    for name, path in config_paths.items():
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    
    # Resolve model path
    configs['robot']["model_path"] = str(Path(pkg_path) / configs['robot']["model_path"])
    return configs['robot'], configs['training'], configs['controller']

def validate_environment_config(robot_config):
    """Validate critical environment parameters"""
    assert robot_config["num_actions"] == 12, "A1 should have 12 actuators"
    assert robot_config["obs_dim"] == 36, "Observation dimension mismatch"
    assert Path(robot_config["model_path"]).exists(), "Model file not found"

# Environment Setup
def make_training_env(robot_config, controller_config, num_envs: int = 4) -> SubprocVecEnv:
    """Create vectorized environments with SubprocVecEnv"""
    def make_env(rank: int) -> Callable[[], LeggedEnv]:
        """Helper function to create environment instances"""
        def _init() -> LeggedEnv:
            # Initialize ROS for each subprocess
            try:
                rclpy.init()
            except RuntimeError:
                pass  # Already initialized
            
            # Create environment with local ROS context
            env = LeggedEnv(robot_config, controller_config)
            return env
            
        return _init
    
    return SubprocVecEnv([make_env(i) for i in range(num_envs)])

# Model Configuration
def build_ppo_model(env, training_config, device="auto"):
    """Initialize PPO model with config parameters"""
    # Auto-detect appropriate device for MLP policies
    policy_type = training_config["ppo_params"]["policy"]
    if "MlpPolicy" in policy_type:
        device = "cpu"
        print(f"Using CPU for {policy_type}")

    return PPO(
        policy=policy_type,
        env=env,
        learning_rate=float(training_config["ppo_params"]["learning_rate"]),
        n_steps=training_config["ppo_params"]["n_steps"],
        batch_size=training_config["ppo_params"]["batch_size"],
        n_epochs=training_config["ppo_params"]["n_epochs"],
        gamma=training_config["ppo_params"]["gamma"],
        gae_lambda=training_config["ppo_params"]["gae_lambda"],
        clip_range=training_config["ppo_params"]["clip_range"],
        ent_coef=training_config["ppo_params"]["ent_coef"],
        verbose=1,
        tensorboard_log=training_config["ppo_params"]["tensorboard_log"],
        device=device
    )

def setup_callbacks(training_config):
    """Create and return list of callbacks"""
    checkpoint_cb = CheckpointCallback(
        save_freq=training_config["checkpoint"]["save_freq"],
        save_path=training_config["checkpoint"]["save_path"],
        name_prefix=training_config["checkpoint"]["name_prefix"]
    )
    return [checkpoint_cb]

# Add new evaluation function
def evaluate_model(model_path, robot_config, controller_config, num_episodes=5):
    """Evaluate a trained model with rendering enabled"""
    rclpy.init()
    
    # Enable rendering for evaluation
    eval_robot_config = robot_config.copy()
    eval_robot_config["render"] = True
    
    print(f"\nEvaluating model: {model_path}")
    print("Starting visualization... (5 episodes)")
    
    # Create evaluation environment
    env = make_training_env(eval_robot_config, controller_config)
    model = PPO.load(model_path, env=env)
    
    # Evaluation metrics
    total_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            step_count += 1
            
            if done:
                break
                
        total_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        print(f"Episode {ep+1}: Reward: {ep_reward:.1f}, Steps: {step_count}")
    
    env.close()
    rclpy.shutdown()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Total Frames Simulated: {sum(episode_lengths)}")

# Modify main training function and add argument parsing
def train_agent():
    """Main training workflow"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or evaluate A1 policy')
    parser.add_argument('--eval', action='store_true', 
                       help='Run evaluation instead of training')
    parser.add_argument('--model-path', type=str,
                       help='Path to model zip file for evaluation')
    parser.add_argument('--continue-training', type=str,
                       help='Path to checkpoint to continue training from')
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            raise ValueError("Must provide --model-path for evaluation")
            
        # Load configurations (without validation for render mode)
        robot_config, _, controller_config = load_configs()
        evaluate_model(args.model_path, robot_config, controller_config)
        return
    
    # Original training workflow below...
    rclpy.init()
    
    # Load and validate configurations
    robot_config, training_config, controller_config = load_configs()
    validate_environment_config(robot_config)

    # Get number of environments from config
    num_envs = training_config.get("num_envs", 4)
    
    # Environment setup with multiple envs
    env = make_training_env(
        robot_config, 
        controller_config,
        num_envs=num_envs
    )
    
    # Add automatic environment normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Model initialization
    if args.continue_training:
        print(f"Continuing training from: {args.continue_training}")
        model = PPO.load(
            args.continue_training,
            env=env,
            device=training_config["ppo_params"].get("device", "auto")
        )
    else:
        model = build_ppo_model(env, training_config, 
                              device=training_config["ppo_params"].get("device", "auto"))
    
    # Removed logger initialization
    callbacks = setup_callbacks(training_config)

    # GPU diagnostics
    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Training execution
    try:
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callbacks,
            tb_log_name="ppo_a1",
            progress_bar=True
        )
        model.save("a1_policy_final_ppo")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    train_agent() 