import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from ament_index_python import get_package_share_directory
import os
from mujoco import viewer
from std_msgs.msg import String

class MujocoSimulator(Node):
    def __init__(self, model_path):
        super().__init__('mujoco_simulator')
        # Set asset directory via environment variable
        os.environ['MUJOCO_ASSETSDIR'] = os.path.join(
            get_package_share_directory('legged_rl_control'),
            'config/assets'
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # ROS 2 interfaces
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.cmd_sub = self.create_subscription(
            JointState, 'joint_commands', self.cmd_callback, 10)
        self.timer = self.create_timer(0.001, self.sim_step)
        self._init_joint_mapping()

        # Add MuJoCo visualization
        self.viewer = viewer.launch_passive(self.model, self.data)

    def _init_joint_mapping(self):
        self.joint_map = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_map[name] = i

    def cmd_callback(self, msg):
        # Map ROS joint commands to MuJoCo actuators
        for name, effort in zip(msg.name, msg.effort):
            if name in self.joint_map:
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_{name}")
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = effort

    def publish_sensor_data(self):
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        for name, idx in self.joint_map.items():
            joint_msg.name.append(str(name))
            joint_msg.position.append(float(self.data.qpos[idx]))
            joint_msg.velocity.append(float(self.data.qvel[idx]))
        self.joint_pub.publish(joint_msg)

        # Publish IMU data according to DeepMind's model structure
        imu_msg = Imu()
        imu_msg.header.frame_id = "trunk"  # Official model uses 'trunk' as base frame
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Update sensor names to match XML definitions
        imu_msg.linear_acceleration.x = float(self.data.sensor('imu_acc').data[0])
        imu_msg.linear_acceleration.y = float(self.data.sensor('imu_acc').data[1])
        imu_msg.linear_acceleration.z = float(self.data.sensor('imu_acc').data[2])
        
        imu_msg.angular_velocity.x = float(self.data.sensor('imu_gyro').data[0])
        imu_msg.angular_velocity.y = float(self.data.sensor('imu_gyro').data[1])
        imu_msg.angular_velocity.z = float(self.data.sensor('imu_gyro').data[2])
        
        # Update orientation sensor name
        imu_msg.orientation.w = float(self.data.sensor('imu_orn').data[0])
        imu_msg.orientation.x = float(self.data.sensor('imu_orn').data[1])
        imu_msg.orientation.y = float(self.data.sensor('imu_orn').data[2])
        imu_msg.orientation.z = float(self.data.sensor('imu_orn').data[3])

        self.imu_pub.publish(imu_msg)

    def sim_step(self):
        mujoco.mj_step(self.model, self.data)
        self.publish_sensor_data()
        self.viewer.sync()

    def advance(self):
        """Update internal state after reset"""
        mujoco.mj_forward(self.model, self.data)

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator(
        model_path=os.path.join(
            get_package_share_directory('legged_rl_control'),
            'config/scene.xml'
        )
    )
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

