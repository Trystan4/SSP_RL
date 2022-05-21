import gym

from stable_baselines3 import DQN
from gym.envs.registration import register
class dqn():
    def __init__(self,environnement, total_e):
        print(environnement)
        if "GridWorld" in environnement:
            #Building the environment, optimum reward_t = 0.74
            register(
                id=environnement,
                entry_point="all_envs.gym_gridworld:GridEnv",
                max_episode_steps=100,
                reward_threshold=0.74,  
            )
        self.env = gym.make(environnement)
        self.total_episodes = total_e
        self.name = "dqn_"+environnement
        self.rewards_episode = []
        self.model = self.model()
    
    def model(self):
        self.model = DQN("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=self.total_episodes, log_interval=4)
        self.model.save(self.name)
        

    def algorithm(self):
        total_r = 0
        del self.model # remove to demonstrate saving and loading
        self.model = DQN.load(self.name)
        self.obs = self.env.reset()
        self.episode = 0
        while self.episode < self.total_episodes:
            action, _states = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, self.done, info = self.env.step(action)
            total_r += reward

            if self.done:
                self.rewards_episode.append(total_r)
                self.episode +=1
                self.obs = self.env.reset()
            
        
        return self.rewards_episode, self.name;