import gym
gym.logger.set_level(40)  # Block warning
env = gym.make('LunarLanderContinuous-v2')
print(env.action_space)
print(env.observation_space.shape)
for i_episode in range(1000):
    observation = env.reset() #初始化环境每次迭代
    for t in range(100):
        env.render() #显示
        print(observation)

        action = env.action_space.sample() #随机选择action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()