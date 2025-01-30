#!/usr/bin/env python3
import argparse
import rclpy
import numpy as np
from stable_baselines3 import SAC
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG
from tqdm import tqdm
import time
from legged_rl_control.utils.logger import EvaluationLogger, plot_evaluation_results
import matplotlib.pyplot as plt
import signal
import sys

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def evaluate_policy(model_path, num_episodes=10, plot=True):
    rclpy.init()
    node = rclpy.create_node('evaluation_node')  # Create proper ROS node
    
    try:
        # Create evaluation config with viewer enabled
        eval_config = A1_CONFIG.copy()
        eval_config.update({
            "launch_viewer": True,
            "visual_evaluation": {
                "enable": True,
                "fps": 60,
                "window_size": (1920, 1080)
            }
        })

        env = LeggedEnv(eval_config)
        
        # Verify viewer initialization
        if env.sim.viewer is None:
            raise RuntimeError("Failed to initialize MuJoCo viewer")

        model = SAC.load(model_path, env=env)
        logger = EvaluationLogger()

        try:
            for ep in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                actions = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    actions.append(action)
                    done = terminated or truncated
                    rclpy.spin_once(node, timeout_sec=0)  # Process ROS callbacks

                # Log episode data
                logger.log_episode({
                    'episode': ep,
                    'reward': episode_reward,
                    'steps': steps,
                    'termination_reason': "timeout" if steps == env._max_episode_steps else info.get('termination_reason', 'other'),
                    'avg_velocity': np.mean(env.sim.data.qvel[:3]),
                    'height_violations': info.get('height_violations', 0),
                    'orientation_violations': info.get('orientation_violations', 0),
                    'actions': actions
                })

        finally:
            env.close()
            node.get_logger().info("Environment closed")

    except Exception as e:
        node.get_logger().error(f"Evaluation failed: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if rclpy.ok():  # Double-check shutdown
            rclpy.shutdown()

    # Generate plots
    if plot:
        plot_evaluation_results()
        print("Saved evaluation plots: evaluation_results.png")
        
    # Add 3D trajectory plot if position tracking is available
    if hasattr(env, 'position_history'):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        positions = np.array(env.position_history)
        ax.plot(positions[:,0], positions[:,1], positions[:,2])
        ax.set_title('Base Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.savefig('trajectory_3d.png')
        plt.close()

    print("\nTermination Reasons:")
    reason_mapping = {
        'excessive_leg_movement': 'Excessive Leg Movement',
        'prolonged_inactivity': 'Prolonged Inactivity',
        # ... other reasons ...
    }
    termination_reasons = env.termination_reasons
    for reason, count in termination_reasons.items():
        print(f"- {reason_mapping.get(reason, reason.capitalize())}: {count}/{num_episodes}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained policy on A1 robot')
    parser.add_argument('model_path', type=str, help='Path to the trained model ZIP file')
    parser.add_argument('-n', '--num_episodes', type=int, default=10, 
                       help='Number of evaluation episodes')
    parser.add_argument('-p', '--plot', action='store_true', help='Generate plots after evaluation')
    args = parser.parse_args()
    
    evaluate_policy(args.model_path, args.num_episodes, args.plot) 