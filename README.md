# SSP_RL

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=sqale_index)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=Trystan4_SSP_RL&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=Trystan4_SSP_RL)

This is a repository related to the Stochastic Shortest Path Problem and its resolutions through the use of RL algorithms (with gym envs).

The goal is to create a simple benchmark implementing several algorithms, which can be controlled and compared for several gym environnments (including 1 created from scratch).

## GymEnvironment

Reproduce Gym Env for a 4x4 (or 8x8) GridWorld with all the particularities of a gym environment.

This is a simple GridWorld whose goal is to go from the beginning to the arrival through the paths without being blocked by the walls.

S -> start
P -> path
L -> Lava
G -> goal

"4x4": ["SPPP", "PLPL", "PPPL", "LPPG"],
    "8x8": [
        "SPPPPPPP",
        "PPPPPPPP",
        "PPPLPPPP",
        "PPPPPLPP",
        "PPPLPPPP",
        "PLLPPPLP",
        "PLPPLPLP",
        "PPPLPPPG",
    ],

## Algorithms

Here are all the algorithms planned with the envs that can be used for now (FrozenLake mean that all implemented gym type environments work).

Sarsa : GridWorld ✔ - FrozenLake ✔  
Q Learning : GridWorld ✔ - FrozenLake  ✔  
Reinforce : GridWorld ✘ - FrozenLake ✘  
DQN : GridWorld ✔ - FrozenLake ✔  
MDP : (Comparison arrives soon)

## Benchmark

You can launch the program by choosing an environment, an algorithm and all the parameters concerning it.

```Bash
git clone https://github.com/Trystan4/SSP_RL.git
cd SSP_RL
py main.py
```
