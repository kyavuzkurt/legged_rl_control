from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get both URDF and MJCF paths
    a1_desc_path = get_package_share_directory('a1_description')
    urdf_path = os.path.join(a1_desc_path, 'urdf', 'robot.urdf')
    mjcf_path = os.path.join(
        get_package_share_directory('legged_rl_control'),
        'config/scene.xml'
    )
    
    return LaunchDescription([
        Node(
            package='legged_rl_control',
            executable='mujoco_sim',
            name='a1_simulator',
            output='screen',
            parameters=[{
                'model_path': mjcf_path,
                'launch_viewer': True  # Enable viewer for evaluation
            }]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='a1_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_path).read()}]
        )
    ]) 