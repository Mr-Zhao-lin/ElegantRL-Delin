import gym
env = gym.make('CartPole-v0')
print(env.action_space)
for i_episode in range(0):
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