from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points


register(
    id="GridWorld-v0",
    entry_point="gym_gridworld.envs:GridEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.74,  # optimum = 0.74
)

register(
    id="GridWorld8x8-v0",
    entry_point="gym_gridworld.envs:GridEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.91,  # optimum = 0.91
)