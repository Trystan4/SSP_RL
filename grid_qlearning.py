import gym
import numpy as np


# Actions in GridWorld environnment :
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


def ChooseAction(Q,etat,espace,epsilon):
    aOpt=np.argmax(np.random.shuffle(Q[etat]))
    if (np.random.random() > epsilon):
        return int(aOpt)
    else:
       action=espace.sample()
       while (action==aOpt):
           action=espace.sample()
       return int(action)
    
gym.envs.register(
    id="GridWorld-v0",
    entry_point="all_envs.gym_gridworld.envs:GridEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.74,  # optimum = 0.74
)
environ= gym.make("GridWorld-v0")
beta=0.5


AS=environ.action_space
observation=environ.reset()

' Tableau Q(s,a)'
Q=np.zeros((environ.observation_space.n,environ.action_space.n))
' Tableau N(s,a)'
N=np.zeros((environ.observation_space.n,environ.action_space.n))


episode=0
epsilon=0.5

while (episode < 10000):
    observation=environ.reset()
    termine=False
    episode=episode+1
    print("debut episode ",episode)
    while (not termine):
        obsC=observation
        #print("Etat courant",obsC)
        a=ChooseAction(Q,obsC,AS,epsilon)
        print("action",a)
        (observation,gain,termine,debug)=environ.step(a)
        #print("Etat suivant",observation)
        ' mise a jour alpha'
        N[obsC,a]=int(N[obsC,a]+1)
        alpha=1/N[obsC,a]
        ' recuperation action optimale'
        aOpt=np.argmax(Q[obsC])
        #print("action Optimale",aOpt)
        ' mise a jour Q table'
        Q[obsC,a]=(1-alpha)*Q[obsC,a]+ alpha*(gain +beta*Q[observation,aOpt])
        #print("terminee",termine)
        #environ.render()
        #print(" ")
    #print("fin de l'episode",episode)
print("fin de la simulation")

print("Q table")
print(Q)

print("Construction de la politique")
pi=np.zeros(environ.observation_space.n)

for i in range(environ.observation_space.n):
    pi[i]=int(np.argmax(Q[i]))

print("politique=",pi)    
    
    