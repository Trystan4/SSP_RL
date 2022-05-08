#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import sys

from benchmark.algorithms.q_learning import grid_qlearning
from benchmark.algorithms.sarsa import grid_sarsa_v1

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
    
    if(int(algo) == 0) :
        beta = float(input("Variable bêta :\n"))  # often 0.5
        epsilon = float(input("Variable epsilon :\n")) # often 0.5
        max_episodes = int(input("Combien de fois voulez vous lancer l'environnement?\n"))
        
        qln_algo = grid_qlearning.Qlearning(environnement[choice_env], beta, epsilon, max_episodes)
        q_table, pi, rewards = qln_algo.algorithm()
        
        print("Q table Q Learning :\n", q_table)
        print("politique = ",pi)
        
        plt.plot(rewards)
        plt.show()
        
    elif(int(algo) == 1):
        total_episodes = int(input("Combien de fois voulez vous lancer l'environnement?\n"))
        max_steps = int(input("Maximum d'actions par épisode : (max 100)\n"))
        epsilon = float(input("Variable epsilon :\n")) # often 0.9
        alpha = float(input("Variable alpha :\n")) # often 0.85
        gamma = float(input("Variable gamma :\n")) # often 0.95
        
        sarsa_algo = grid_sarsa_v1.sarsa(environnement[choice_env], epsilon, total_episodes, max_steps, alpha, gamma )
        q_table, performance, rewards = sarsa_algo.algorithm()
        
        #Visualizing the Q-matrix
        print("Q table Sarsa : \n",q_table)
        print("Performance :", performance)
        
        plt.plot(rewards)
        plt.show()
        
    elif(int(algo) == 2):
        print(NOT_IMPLEMENTED)
    elif(int(algo) == 3):
        print(NOT_IMPLEMENTED)
    else:
        print("algorithme non valable")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())    