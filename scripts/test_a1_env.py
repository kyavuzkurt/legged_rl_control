import rclpy
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG

def test_environment():
    rclpy.init()
    
    try:
        env = LeggedEnv(A1_CONFIG)
        
        # Test observation space
        obs = env.reset()
        assert obs.shape == (A1_CONFIG["obs_dim"],), "Observation shape mismatch"
        
        # Test action space
        action = env.action_space.sample()
        assert action.shape == (A1_CONFIG["num_actions"],), "Action shape mismatch"
        
        # Test step function
        obs, reward, done, info = env.step(action)
        print(f"First step successful! Reward: {reward}, Observation: {obs[:3]}...")
        
        # Test multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                break
                
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    test_environment() 