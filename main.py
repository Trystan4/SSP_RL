#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import sys
import torch
import zipfile

from benchmark.algorithms.q_learning import grid_qlearning
from benchmark.algorithms.sarsa import grid_sarsa_v1
from benchmark.algorithms.dqn import grid_dqn

NOT_IMPLEMENTED = "pas encore implémenté"
environnement = {
    0 : "GridWorld-v0",
    1 : "FrozenLake-v1",
    2 : "FrozenLake8x8-v1"
}

def main():
    print("Quel environnement voulez-vous?")
    choice_env = int(input("0: GridWorld-v0 \n1: FrozenLake-v1 \n2: FrozenLake8x8-v1\n"))
    
    print("Quel(s) algorithmes voulez vous lancer?")
    algo = input("0: Qlearning \n1: Sarsa \n2: DQN \n3: Reinforce\n")
    max_episodes = int(input("Combien de fois voulez vous lancer l'environnement?\n"))
    epsilon = float(input("Variable epsilon (0.5 q learning / 0.9 sarsa) :\n")) # often 0.5 for QLN and 0.9 for SARSA
    
    if(int(algo) == 0 or int(algo) == -1) :
        beta = float(input("Variable bêta (0.5) :\n"))  # often 0.5
        
        qln_algo = grid_qlearning.Qlearning(environnement[choice_env], beta, epsilon, max_episodes)
        q_table, qln_pi = qln_algo.algorithm()
        q_rewards_episode = qln_algo.simulation()
        print("Q table Q Learning :\n", q_table)
        print("politique QLN = ",qln_pi)
        
        plt.plot(q_rewards_episode)
        plt.savefig("Q_Learning_rewards")
        
    if(int(algo) == 1 or int(algo) == -1): # SARSA
        max_steps = int(input("Maximum d'actions par épisode : (max 100)\n"))
        alpha = float(input("Variable alpha (0.85) :\n")) # often 0.85
        gamma = float(input("Variable gamma (0.95) :\n")) # often 0.95
        
        sarsa_algo = grid_sarsa_v1.sarsa(environnement[choice_env], epsilon, max_episodes, max_steps, alpha, gamma )
        q_table, sarsa_pi, performance = sarsa_algo.algorithm()
        sarsa_rewards_epsiode = sarsa_algo.simulation()
        #Visualizing the Q-matrix
        print("Q table Sarsa : \n",q_table)
        print("politique Sarsa = ",sarsa_pi)
        print("Performance :", performance)
        
        plt.plot(sarsa_rewards_epsiode)
        plt.savefig("SARSA_rewards")
        
    if(int(algo) == 2 or int(algo) == -1): # Deep Q Learning
        dqn_algo = grid_dqn.dqn(environnement[choice_env], max_episodes)
        dqn_rewards_episode, name = dqn_algo.algorithm()
        plt.plot(dqn_rewards_episode)
        plt.savefig("DQN_rewards")
        zip_n = name + ".zip"
        with zipfile.ZipFile(zip_n,"r") as zip_ref:
            zip_ref.extractall(name)
    
        device = torch.device('cpu')
        path_p = name + "/policy.pth"
        path_o = name + "/policy.optimizer.pth"
        path_v = name + "/pytorch_variables.pth"
        policy = torch.load(path_p,  map_location=device )
        policy_opt = torch.load(path_o,  map_location=device )
        torch_var = torch.load(path_v,  map_location=device )
        
        print(policy, policy_opt, torch_var)
    if(int(algo) == 3 or int(algo) == -1): # Reinforce
        print(NOT_IMPLEMENTED)
    elif(int(algo) < -1 or int(algo) > 4):
        print("algorithme non valable")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())    