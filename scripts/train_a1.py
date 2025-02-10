import rclpy
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np
import torch
import os
import argparse

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
def make_training_env(robot_config, controller_config, num_envs=None):
    """Create parallel environments with SubprocVecEnv"""
    num_envs = num_envs or (os.cpu_count() - 1 or 1)
    
    # Use a factory function to ensure proper ROS initialization
    def make_env():
        rclpy.init()  # Initialize ROS context for each subprocess
        env = LeggedEnv(robot_config, controller_config)
        return env
        
    return SubprocVecEnv([make_env for _ in range(num_envs)])

# Model Configuration
def build_sac_model(env, training_config, device='auto'):
    """Configure and return SAC model"""
    sac_params = training_config["sac_params"]
    return SAC(
        policy=sac_params["policy"],
        env=env,
        verbose=1,
        **{k: v for k, v in sac_params.items() if k != "policy"}
    )

# Callbacks & Logging
def setup_callbacks(training_config):
    """Create and return list of callbacks"""
    checkpoint_cb = CheckpointCallback(
        save_freq=training_config["checkpoint"]["save_freq"],
        save_path=training_config["checkpoint"]["save_path"],
        name_prefix=training_config["checkpoint"]["name_prefix"]
    )
    
    return [checkpoint_cb]

# Main Training Logic
def train_agent():
    """Main training workflow"""
    robot_config, training_config, controller_config = load_configs()
    validate_environment_config(robot_config)

    env = make_training_env(
        robot_config, 
        controller_config,
        num_envs=training_config.get("num_envs", os.cpu_count() - 1)
    )
    
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    model = build_sac_model(env, training_config, 
                           device=training_config["sac_params"].get("device", "auto"))
    
    callbacks = setup_callbacks(training_config)

    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    try:
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callbacks,
            tb_log_name="sac_a1",
            progress_bar=True
        )
        model.save("a1_policy_final")
    finally:
        env.close()

def evaluate_policy(model_path, num_episodes=5):
    """Evaluate trained policy with visualization"""
    # Initialize ROS only if not already initialized
    if not rclpy.ok():
        rclpy.init()
    
    try:
        # Load config with rendering enabled
        robot_config, _, controller_config = load_configs()
        robot_config["render"] = True
        validate_environment_config(robot_config)

        # Create single environment instance
        env = LeggedEnv(robot_config, controller_config)
        
        try:
            # Load trained model
            model = SAC.load(model_path)
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"Total reward: {total_reward:.2f}")
                print("="*40)
                
        finally:
            env.close()
            
    finally:
        # Ensure final shutdown
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-with-visual', action='store_true',
                       help='Run evaluation with MuJoCo visualization')
    parser.add_argument('--model-path', type=str, default='a1_policy_final',
                       help='Path to trained model zip file')
    args = parser.parse_args()

    if args.eval_with_visual:
        evaluate_policy(args.model_path)
    else:
        train_agent() 