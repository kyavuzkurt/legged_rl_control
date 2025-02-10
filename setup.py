from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'legged_rl_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml', 'setup.cfg']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.xml') + 
         glob('config/**/*.xml', recursive=True)),
        (os.path.join('share', package_name, 'config/assets'),
         glob('config/assets/**/*', recursive=True)),
        (os.path.join('share', package_name, 'config/robots'), 
         glob('config/robots/*.yaml')),
        (os.path.join('share', package_name, 'config/training'), 
         glob('config/training/*.yaml')),
        (os.path.join('share', package_name, 'config/controllers'), 
         glob('config/controllers/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'stable_baselines3',
        'numpy',
        'gym==0.21.0',
        'pybullet',
        'torch',
        'tensorboard',
        'matplotlib',
        'pandas',
        'pyyaml'
    ],
    zip_safe=True,
    maintainer='Kadir Yavuz Kurt',
    maintainer_email='k.yavuzkurt1@gmail.com',
    description='RL-based legged robot control with MuJoCo',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_sim = legged_rl_control.nodes.mujoco_sim:main',
            'policy_node = legged_rl_control.nodes.policy_node:main',
        ],
    },
) 