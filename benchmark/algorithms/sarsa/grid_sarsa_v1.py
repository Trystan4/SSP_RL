import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt


class sarsa():
    def __init__(self, environnement, epsilon, total_episodes, max_steps, alpha, gamma):
        if "GridWorld" in environnement:
            #Building the environment, optimum reward_t = 0.74
            register(
                id=environnement,
                entry_point="all_envs.gym_gridworld:Gridworld",
                max_episode_steps=100,
                reward_threshold=0.74,  
            )
            
        self.env = gym.make(environnement)
        #Defining the different parameters
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        
        #Initializing the Q-matrix
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        #Initializing pi politic
        self.pi = []
        #Initializing the reward
        self.rewards = []
        
    #Function to choose the next action
    def choose_action(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action
    
    #Function to learn the Q-value
    def update(self):
        self.predict = self.Q[self.state1, self.action1]
        self.target = self.reward + self.gamma * self.Q[self.state2, self.action2]
        self.Q[self.state1, self.action1] = self.Q[self.state1, self.action1] + self.alpha * (self.target - self.predict)
        
    def algorithm(self):
        self.reward=0
        #Starting the SARSA learning
        for episode in range(self.total_episodes):
            self.t = 0
            self.state1 = self.env.reset()
            self.action1 = self.choose_action(self.state1)
            print("------------------------")
            print("début episode ", episode)
            print("------------------------")
            print("first action",self.action1)
            while self.t < self.max_steps: 
                #Getting next state
                self.state2, self.reward, done, info = self.env.step(self.action1)
                # print(info)
                # Array for rewards in time (for each episode)
                self.rewards.append(self.reward)
                #Choosing the next action
                self.action2 = self.choose_action(self.state2)
                print("next action", self.action2)
                
                #Learning Q-value
                self.update()
                #action/state update for next step
                self.state1 = self.state2
                self.action1 = self.action2
                
                #update respective values
                self.t +=1
                self.reward += 1
                
                if done:
                    break
        #Evaluating the performance
        self.perf = self.reward/self.total_episodes
        
        for i in range(self.env.observation_space.n):
            self.pi[i]=int(np.argmax(self.Q[i]))
            
        return self.Q, self.perf,self.pi, self.rewards
                







