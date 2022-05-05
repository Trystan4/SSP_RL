import numpy as np
import gym

#Building the environment

# gym.envs.register(
#     id="GridWorld-v0",
#     entry_point="all_envs.gym_gridworld.envs:GridEnv",
#     kwargs={"map_name": "4x4"},
#     max_episode_steps=100,
#     reward_threshold=0.74,  # optimum = 0.74
# )
# environ= gym.make("GridWorld-v0")

env = gym.make('FrozenLake-v1')

#Defining the different parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

#Initializing the Q-matrix
Q = np.zeros((env.observation_space.n, env.action_space.n))

#Function to choose the next action
def choose_action(state):
	action=0
	if np.random.uniform(0, 1) < epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmax(Q[state, :])
	return action

#Function to learn the Q-value
def update(state, state2, reward, action, action2):
	predict = Q[state, action]
	target = reward + gamma * Q[state2, action2]
	Q[state, action] = Q[state, action] + alpha * (target - predict)

#Initializing the reward
reward=0

#Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)
    print("------------------------")
    print("début episode ", episode)
    print("------------------------")
    print("first action",action1)
    while t < max_steps: 
        #Getting next state
        state2, reward, done, info = env.step(action1)
        
        #Choosing the next action
        action2 = choose_action(state2)
        print("next action", action2)
        
        #Learning Q-value
        update(state1, state2, reward, action1, action2)
        #action/state update for next step
        state1 = state2
        action1 = action2
        
        #update respective values
        t +=1
        reward += 1
        
        if done:
            break
#Evaluating the performance
print("Performance : ", reward/total_episodes)

#Visualizing the Q-matrix
print(Q)
