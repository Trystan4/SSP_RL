import gym

from stable_baselines3 import DQN

env = gym.make("FrozenLake-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_frozenlake")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_frozenlake")

obs = env.reset()
episodes = 0
while episodes < 10:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        episodes +=1
        obs = env.reset()