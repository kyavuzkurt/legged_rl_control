import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class TrainingLogger:
    def __init__(self, log_dir="logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = self.log_dir / "training_log.csv"
        
        # Initialize CSV file
        with open(self.train_log, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total_timesteps', 'episode_reward_mean',
                'episode_len_mean', 'value_loss', 'policy_loss',
                'entropy_coeff', 'alpha', 'grad_norm'
            ])

    def log_training(self, data):
        with open(self.train_log, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                data['total_timesteps'],
                data.get('episode_reward_mean', 0),
                data.get('episode_len_mean', 0),
                data.get('value_loss', 0),
                data.get('policy_loss', 0),
                data.get('entropy_coeff', 0),
                data.get('alpha', 0),
                data.get('grad_norm', 0)
            ])

class EvaluationLogger:
    def __init__(self, log_dir="logs/evaluation"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_log = self.log_dir / "evaluation_log.csv"
        
        with open(self.eval_log, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'episode', 'reward', 'steps',
                'termination_reason', 'avg_velocity',
                'height_violations', 'orientation_violations',
                'action_mean', 'action_std'
            ])

    def log_episode(self, episode_data):
        with open(self.eval_log, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                episode_data['episode'],
                episode_data['reward'],
                episode_data['steps'],
                episode_data['termination_reason'],
                episode_data['avg_velocity'],
                episode_data['height_violations'],
                episode_data['orientation_violations'],
                np.mean(episode_data['actions']),
                np.std(episode_data['actions'])
            ])

def plot_training_curves(log_path="logs/training/training_log.csv"):
    data = np.genfromtxt(log_path, delimiter=',', names=True)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    
    # Reward and Length
    axs[0,0].plot(data['total_timesteps'], data['episode_reward_mean'])
    axs[0,0].set_title('Training Rewards')
    axs[0,0].set_ylabel('Mean Episode Reward')
    
    axs[0,1].plot(data['total_timesteps'], data['episode_len_mean'])
    axs[0,1].set_title('Episode Lengths')
    axs[0,1].set_ylabel('Mean Steps per Episode')
    
    # Losses
    axs[1,0].plot(data['total_timesteps'], data['value_loss'], label='Value Loss')
    axs[1,0].plot(data['total_timesteps'], data['policy_loss'], label='Policy Loss')
    axs[1,0].set_title('Loss Curves')
    axs[1,0].legend()
    
    # Entropy and Grad Norm
    axs[1,1].plot(data['total_timesteps'], data['entropy_coeff'])
    axs[1,1].set_title('Entropy Coefficient')
    
    axs[2,0].plot(data['total_timesteps'], data['grad_norm'])
    axs[2,0].set_title('Gradient Norm')
    
    # Alpha
    axs[2,1].plot(data['total_timesteps'], data['alpha'])
    axs[2,1].set_title('Temperature (α)')
    
    for ax in axs.flat:
        ax.set(xlabel='Timesteps')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_evaluation_results(log_path="logs/evaluation/evaluation_log.csv"):
    data = np.genfromtxt(log_path, delimiter=',', names=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward Distribution
    axs[0,0].hist(data['reward'], bins=20)
    axs[0,0].set_title('Reward Distribution')
    axs[0,0].set_xlabel('Total Reward')
    axs[0,0].set_ylabel('Frequency')
    
    # Termination Reasons
    reasons, counts = np.unique(data['termination_reason'], return_counts=True)
    axs[0,1].bar(reasons.astype(str), counts)
    axs[0,1].set_title('Termination Reasons')
    
    # Velocity vs Actions
    axs[1,0].scatter(data['avg_velocity'], data['action_mean'])
    axs[1,0].set_title('Average Velocity vs Mean Action')
    axs[1,0].set_xlabel('Avg Velocity (m/s)')
    axs[1,0].set_ylabel('Mean Action Magnitude')
    
    # Action Distribution
    axs[1,1].hist(data['action_std'], bins=20)
    axs[1,1].set_title('Action Standard Deviation Distribution')
    axs[1,1].set_xlabel('Action STD')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close() 