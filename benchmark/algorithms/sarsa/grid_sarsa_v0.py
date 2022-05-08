import gym
import numpy as np


# Actions in GridWorld environnment :
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


def choose_action(Q,etat,espace,epsilon):
    a_opt=np.argmax(np.random.shuffle(Q[etat]))
    if (np.random.random() > epsilon):
        return int(a_opt)
    else:
       action=espace.sample()
       while (action==a_opt):
           action=espace.sample()
       return int(action)
    
# gym.envs.register(
#     id="GridWorld-v0",
#     entry_point="all_envs.gym_gridworld.envs:GridEnv",
#     kwargs={"map_name": "4x4"},
#     max_episode_steps=100,
#     reward_threshold=0.74,  # optimum = 0.74
# )
# environ= gym.make("GridWorld-v0")
environ = gym.make("FrozenLake-v1")



AS=environ.action_space
observation=environ.reset()

' Tableau q_table(s,a)'
q_table=np.zeros((environ.observation_space.n,environ.action_space.n))
' Tableau N(s,a)'
N=np.zeros((environ.observation_space.n,environ.action_space.n))


episode=0
epsilon=0.9
beta=0.5

while (episode < 10000):
    observation=environ.reset()
    termine=False
    episode=episode+1
    print("debut episode ",episode)
    while (not termine):
        obsC=observation
        #print("Etat courant",obsC)
        a=choose_action(q_table,obsC,AS,epsilon)
        print("action",a)
        (observation,gain,termine,debug)=environ.step(a)
        #print("Etat suivant",observation)
        ' mise a jour alpha'
        N[obsC,a]=int(N[obsC,a]+1)
        alpha=1/N[obsC,a]
        ' recuperation action optimale'
        a_opt=np.argmax(q_table[obsC])
        #print("action Optimale",a_opt)
        ' mise a jour q_table table pour sarsa'
        q_table[obsC,a]= Q[obsC,a] + alpha * (gain + beta * (Q[observation,a_opt] - Q[obsC,a]))
        
        #environ.render()
print("fin de la simulation")

print("Q table SARSA")
print(Q)

print("Construction de la politique")
pi=np.zeros(environ.observation_space.n)

for i in range(environ.observation_space.n):
    pi[i]=int(np.argmax(Q[i]))

print("politique=",pi)    
    
    