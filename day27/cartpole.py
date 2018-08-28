import gym

if __name__ == "__main__":
    # get instance of the environment
    env = gym.make("CartPole-v0")
    # starting with initial reward as ZERO
    total_reward = 0.0
    # total steps intialized to ZERO
    total_steps = 0
    # getting random sate of the environment
    obs = env.reset()

    while True:
        # choosing random action
        action = env.action_space.sample()
        # taking a step with that action and getting new observed state vriables
        obs, reward, done, _ = env.step(action)
        # adding to global reward
        total_reward += reward
        total_steps += 1
        if done:
            break
    print ("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))