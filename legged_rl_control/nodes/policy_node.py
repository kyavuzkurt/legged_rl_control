import rclpy
import numpy as np
import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from legged_rl_control.rl.policies import SACPolicy

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.policy = SACPolicy.load("legged_sac")
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.callback, 10)
        self.cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)

    def callback(self, msg):
        obs = self._msg_to_obs(msg)
        action = self.policy.predict(obs)
        self._publish_action(action)

    def _msg_to_obs(self, msg):
        return np.concatenate([
            msg.position,
            msg.velocity,
            msg.effort
        ])

    def _publish_action(self, action):
        cmd_msg = JointState()
        cmd_msg.effort = action.tolist()
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 