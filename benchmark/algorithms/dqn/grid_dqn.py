import gym

from stable_baselines3 import DQN

class dqn():
    def __init__(self,environnement, total_e):
        self.env = gym.make(environnement)
        self.total_episodes = total_e
        self.name = "dqn_"+environnement
        self.model = self.model()
    
    def model(self):
        self.model = DQN("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=10000, log_interval=4)
        self.model.save(self.name)
        

    def algorithm(self):
        del self.model # remove to demonstrate saving and loading
        self.model = DQN.load(self.name)
        self.obs = self.env.reset()
        self.episode = 0
        while self.episode < self.total_episodes:
            action, _states = self.model.predict(self.obs, deterministic=True)
            self.obs, self.reward, self.done, info = self.env.step(action)
            self.env.render()
            if self.done:
                self.episode +=1
                self.obs = self.env.reset()