import gym
import numpy as np


# Actions in GridWorld environnment :
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

class Qlearning():
    def __init__(self, environnement,beta,epsilon, max_episodes):
        # if(environnement == "GridWorld-v0"):
        #     gym.envs.register(
        #         id=environnemen,
        #         entry_point="all_envs.gym_gridworld:GridEnv",
        #         kwargs={"map_name": "4x4"},
        #         max_episode_steps=100,
        #         reward_threshold=0.74,  # optimum = 0.74
        #     )
        #     environ= gym.make("GridWorld-v0")
        # else :
        self.environ = gym.make(environnement)
        self.beta = beta
        self.epsilon = epsilon
        self.max_e = max_episodes
        self.AS=self.environ.action_space
        self.observation=self.environ.reset()

        ' Tableau Q(s,a)'
        self.Q=np.zeros((self.environ.observation_space.n,self.environ.action_space.n))
        ' Tableau N(s,a)'
        self.N=np.zeros((self.environ.observation_space.n,self.environ.action_space.n))
        self.rewards=[]
        self.episode=0
        
    def choose_action(self, etat):
        aOpt=np.argmax(np.random.shuffle(self.Q[etat]))
        if (np.random.random() > self.epsilon):
            return int(aOpt)
        else:
            action=self.AS.sample()
        while (action==aOpt):
            action=self.AS.sample()
        return int(action)
    
    def algorithm(self):
        while (self.episode < self.max_e):
            observation=self.environ.reset()
            termine=False
            self.episode=self.episode+1
            print("debut episode ",self.episode)
            while (not termine):
                obs_c=observation
                #print("Etat courant",obs_c)
                a=self.choose_action(obs_c)
                print("action",a)
                (observation,gain,termine,debug)=self.environ.step(a)
                #print("Etat suivant",observation)
                ' mise a jour alpha'
                self.N[obs_c,a]=int(self.N[obs_c,a]+1)
                alpha=1/self.N[obs_c,a]
                ' recuperation action optimale'
                aOpt=np.argmax(self.Q[obs_c])
                #print("action Optimale",aOpt)
                ' mise a jour Q table'
                self.Q[obs_c,a]=(1-alpha)*self.Q[obs_c,a]+ alpha*(gain +self.beta*self.Q[observation,aOpt])
                
                # tableau des gains au fur et à mesure des épisodes
                self.rewards.append(gain)
            #print("fin de l'episode",episode)
        print("fin de la simulation")
        
        print("Construction de la politique")
        self.pi=np.zeros(self.environ.observation_space.n)

        for i in range(self.environ.observation_space.n):
            self.pi[i]=int(np.argmax(self.Q[i]))

        return self.Q, self.pi, self.rewards