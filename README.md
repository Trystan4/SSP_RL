# SSP_RL

This is a repository related to the SSP problem and its resolutions through the use of RL algorithms (with gym envs).

The goal is to create a simple benchmark implementing several algorithms, which can be controlled and compared for several gym environnments (including 1 created from scratch).

## GymEnvironment

Reproduce Gym Env for a 4x4 (or 8x8) GridWorld with all the particularities of a gym environment.

## Algorithms

Here are all the algorithms planned with the envs that can be used for now (FrozenLake mean that all implemented gym type environments work).

Sarsa : Grid ✘ - FrozenLake ✔
Q Learning : Grid ✘ - FrozenLake  ✔
Reinforce : Grid ✘ - FrozenLake ✘
DQN : Grid ✘ - FrozenLake ✘
MDP : Grid ✘ - FrozenLake ✘

## Benchmark

You can launch the program by choosing an environment, an algorithm and all the parameters concerning it.

```Bash
git clone https://github.com/Trystan4/SSP_RL.git
cd SSP_RL
py main.py
```

## Comparison

will arrive later !
