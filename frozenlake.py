import gym
import numpy as np
import random
import subprocess as sp
import time

env = gym.make("FrozenLake-v0")


LEARNING_RATE = 0.2
DISCOUNT_RATE = 0.95
EPISODES = 10000
STEPS_PER_EPISODE = 100

EPSILON_RATE = 1
EPSILON_DECAY_RATE = 0.01
MIN_EPSILON = 0.01
MAX_EPSILON = 1

REWARDS_ALL_EPISODES = []

ACTION_SPACE_SIZE = env.action_space.n
STATE_SPACE_SIZE = env.observation_space.n

q_table = np.zeros((STATE_SPACE_SIZE,ACTION_SPACE_SIZE))

for episode in range(EPISODES):
	CURRENT_STATE = env.reset()
	done = False
	CURRENT_EPISODE_REWARD = 0

	for step in range(STEPS_PER_EPISODE):

		THRESHOLD = random.uniform(0, 1)
		if THRESHOLD > EPSILON_RATE:
			action = np.argmax(q_table[CURRENT_STATE,:])
		else:
			action = env.action_space.sample()

		NEW_STATE, reward, done, _ = env.step(action)

		q_table[CURRENT_STATE, action] = q_table[CURRENT_STATE, action] * (1 - LEARNING_RATE) + \
		LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[NEW_STATE, :]))

		CURRENT_STATE = NEW_STATE
		CURRENT_EPISODE_REWARD += reward

		if done:
			break

		EPSILON_RATE = MIN_EPSILON + \
		(MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY_RATE*episode)


	REWARDS_ALL_EPISODES.append(CURRENT_EPISODE_REWARD)


for episode in range(5):
	CURRENT_STATE = env.reset()
	done = False
	print("*****EPISODE ", episode+1, "*****\n\n\n\n")
	time.sleep(1)

	for step in range(STEPS_PER_EPISODE):        
		tmp = sp.call('clear',shell=True)
		env.render()
		time.sleep(0.3)
		
		action = np.argmax(q_table[CURRENT_STATE,:])        
		NEW_STATE, reward, done, _ = env.step(action)

		if done:
			tmp = sp.call('clear',shell=True)
			env.render()
			if reward == 1:
				print("****You reached the goal!****")
				time.sleep(3)
			else:
				print("****You fell through a hole!****")
				time.sleep(3)
				tmp = sp.call('clear',shell=True)
			break

		CURRENT_STATE = NEW_STATE


env.close()
print(REWARDS_ALL_EPISODES)
print(q_table)



