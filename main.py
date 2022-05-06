#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
from benchmark.algorithms.q_learning import grid_qlearning
from benchmark.algorithms.sarsa import grid_sarsa_v1

NOT_IMPLEMENTED = "pas encore implémenté"


if __name__ == "__main__":
    print("Quel environnement? (GridWorld-v0, FrozenLake-v1, FrozenLake8x8-v1) :")
    environnement = input()
    print("Quel(s) algorithmes voulez vous lancer? (Qlearning, SARSA, DQN, Reinforce)")
    algo = input()
    if(algo.lower() == "qlearning") :
        print("Variable bêta :")
        beta = float(input())  # often 0.5
        print("Variable epsilon :")
        epsilon = float(input()) # often 0.5
        print("Combien de fois voulez vous lancer l'environnement?")
        max_episodes = int(input())
        
        qln_algo = grid_qlearning.Qlearning(environnement, beta, epsilon, max_episodes)
        q_table, pi, rewards = qln_algo.algorithm()
        print("Q table Q Learning")
        print(q_table)
        print("politique=",pi)
        
        plt.plot(rewards)
        plt.show()
        
    elif(algo.lower() == "sarsa"):
        
        print("Combien de fois voulez vous lancer l'environnement?")
        total_episodes = int(input())
        print("Maximum d'actions par épisode : (max 100)")
        max_steps = int(input())
        print("Variable epsilon :")
        epsilon = float(input()) # often 0.9
        print("Variable alpha :")
        alpha = float(input()) # often 0.85
        print("Variable gamma :")
        gamma = float(input()) # often 0.95
        
        sarsa_algo = grid_sarsa_v1.sarsa(environnement, epsilon, total_episodes, max_steps, alpha, gamma )
        q_table, performance, rewards = sarsa_algo.algorithm()
        #Visualizing the Q-matrix
        print("Q table Sarsa : \n",q_table)
        print("Performance :", performance)
        plt.plot(rewards[:100])
        plt.show()
        
    elif(algo.lower() == "dqn"):
        print(NOT_IMPLEMENTED)
    elif(algo.lower() == "reinforce"):
        print(NOT_IMPLEMENTED)
    else:
        print("algorithme non valable")
    