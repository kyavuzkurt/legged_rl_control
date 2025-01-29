from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config',
            default_value='a1',
            description='Robot configuration name'
        ),
        Node(
            package='legged_rl_control',
            executable='mujoco_sim',
            name='mujoco_simulator',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path')
            }]
        )
    ]) 