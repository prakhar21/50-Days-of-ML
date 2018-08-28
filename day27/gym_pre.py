import gym

ENV = 'CartPole-v0'
environment = gym.make(ENV)

observation = environment.reset()
print observation #returns [x-position, speed, angle, angular velocity]

print environment.action_space #action space  {0 : left, 1 : right}
print environment.observation_space #observation space vec[-inf, inf] having 4 values

print environment.step(0) #(observation state, reward, done?, extra information)

print environment.action_space.sample() #random action [0, 1]
print environment.observation_space.sample() #random state vec[4]