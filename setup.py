from setuptools import setup

setup(name="gym_gridworld",
      version="0.1",
      url="https://github.com/Trystan4/SSP_RL",
      author="Trystan Roches",
      packages = ["all_envs.gym_gridworld.envs", "benchmark.algorithms", "wo_gym"],
      install_requires = ["gym", "numpy", "matplotlib", "contextlib2", "typing", "stable-baselines3", "torch"]
)