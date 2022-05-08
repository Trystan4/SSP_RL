import gym
import numpy as np


# Actions in GridWorld environnment :
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


def choose_action(q_table,etat,espace,epsilon):
    a_opt=np.argmax(np.random.shuffle(q_table[etat]))
    if (np.random.random() > epsilon):
        return int(a_opt)
    else:
       action=espace.sample()
       while (action==a_opt):
           action=espace.sample()
       return int(action)
    
    
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
        q_table[obsC,a]= q_table[obsC,a] + alpha * (gain + beta * (q_table[observation,a_opt] - q_table[obsC,a]))
        
        #environ.render()
print("fin de la simulation")

print("q_table table SARSA")
print(q_table)

print("Construction de la politique")
pi=np.zeros(environ.observation_space.n)

for i in range(environ.observation_space.n):
    pi[i]=int(np.argmax(q_table[i]))

print("politique=",pi)    
    
    