import rclpy
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG

def test_environment():
    rclpy.init()
    env = None  # Initialize env here
    
    try:
        # Create modified config for evaluation
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
        
        # Test observation space
        obs, _ = env.reset()
        print(f"Actual observation shape: {obs.shape}")  # Debug output
        assert obs.shape == (A1_CONFIG["obs_dim"],), f"Observation shape mismatch: {obs.shape} vs expected {A1_CONFIG['obs_dim']}"
        
        # Test action space
        action = env.action_space.sample()
        assert action.shape == (A1_CONFIG["num_actions"],), "Action shape mismatch"
        
        # Test step function
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"First step successful! Reward: {reward}, Observation: {obs[:3]}...")
        
        # Test multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
    finally:
        if env is not None:
            env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    test_environment() 