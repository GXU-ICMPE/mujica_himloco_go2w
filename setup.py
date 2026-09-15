from setuptools import find_packages, setup

setup(
    name='mujica-himloco-go2w',
    version='0.2.0',
    author='Junfeng Long, Zirui Wang, Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(),
    package_data={'mujica.isaaclab': ['defaults.json']},
    author_email='',
    description='MUJICA PPO for Go2W with native Isaac Lab and legacy Isaac Gym backends',
    # Isaac Lab/Sim and CUDA torch are provisioned by the existing simulator
    # environment. This package must not install isaacgym or replace rsl-rl.
    install_requires=['numpy', 'scipy', 'matplotlib', 'tensorboard'],
)
