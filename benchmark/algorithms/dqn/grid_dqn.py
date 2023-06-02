import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gym.envs.registration import register

class dqn():
    def __init__(self,environnement, total_e, total_epoch, gamma):
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
        self.epochs = total_epoch
        self.name = "dqn_"+environnement
        self.gamma = gamma
        self.rewards_episode = []
        self.model = self.model()
    
    def model(self):
        self.model = DQN("MlpPolicy", self.env, verbose=0, buffer_size=100000, learning_starts=10000 ,train_freq=1, 
            batch_size=25, gamma=self.gamma, policy_kwargs={'net_arch': [64,64]}, exploration_fraction=0.1, target_update_interval=1000)
        
        self.model.learn(total_timesteps=self.total_episodes, n_eval_episodes=self.epochs, log_interval=100)
        self.model.save(self.name)
        

    def algorithm(self):
        self.model = DQN.load(self.name)
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=self.total_episodes, deterministic=True)
        print(mean_reward, std_reward)
        self.episode = 0
        self.rewards_episode = []
        while self.episode < self.total_episodes:
            obs = self.env.reset()
            done = False
            self.episode +=1
            total_r = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_r = total_r + reward
            self.rewards_episode.append(total_r)
            
        return self.rewards_episode, self.name