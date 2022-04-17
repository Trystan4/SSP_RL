from gym_gridworld.envs.GridWorld import GridWorld4x4
import matplotlib.pyplot as plt
states_n = 16
actions_n = 4

# Set learning parameters
lr = 0.5
y = .99
num_episodes = 100000

Q, cumul_reward_list = GridWorld4x4.q_learning(GridWorld4x4, states_n, actions_n, lr, y, num_episodes)

print("Score over time: " +  str(sum(cumul_reward_list[-100:])/100.0))
# Vue de la Q-Table après 10 000 épisodes

print(Q)
plt.plot(cumul_reward_list[:10])
plt.ylabel('Cumulative reward')
plt.xlabel('Étape')
plt.show()