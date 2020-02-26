import gym
import random
import numpy as np
import subprocess as sp
import time

env = gym.make('Taxi-v3')

ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.n

#print(ACTION_SPACE,OBSERVATION_SPACE)
#print(env.action_space)


q_table = np.zeros((OBSERVATION_SPACE,ACTION_SPACE))

LEARNING_RATE = 0.3
DISCOUNT_RATE = 0.9
EPISODES = 10000

EPSILON_RATE = 1
EPSILON_DECAY_RATE = 0.9
MIN_EPSILON = 0.01
MAX_EPSILON = 1



for episode in range(EPISODES):
	CURRENT_STATE = env.reset()
	#print(CURRENT_STATE)
	done = False

	while not done:

		PROBABILTY = random.uniform(0,1)
		if PROBABILTY > EPSILON_RATE:
			action = np.argmax(q_table[CURRENT_STATE,:])
		else:
			action = env.action_space.sample()

		NEW_STATE, reward, done, info = env.step(action)

		q_table[CURRENT_STATE, action] = q_table[CURRENT_STATE, action] * (1 - LEARNING_RATE) + \
		LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[NEW_STATE, :]))

		if done:
			break

		EPSILON_RATE = max(EPSILON_RATE * EPSILON_DECAY_RATE, MIN_EPSILON)

	#print(reward)

for episode in range(5):
	CURRENT_STATE = env.reset()
	done = False
	print("*****EPISODE ", episode+1, "*****\n\n\n\n")
	time.sleep(1)

	while not done:        
		tmp = sp.call('clear',shell=True)
		env.render()
		time.sleep(0.5)
		
		action = np.argmax(q_table[CURRENT_STATE,:])        
		NEW_STATE, reward, done, _ = env.step(action)

		if done:
			tmp = sp.call('clear',shell=True)
			env.render()
			if reward == 1:
				print("****Good Job!****")
				time.sleep(5)
			else:
				print("****You Failed!****")
				time.sleep(5)
				tmp = sp.call('clear',shell=True)
			break

		CURRENT_STATE = NEW_STATE


print(q_table)

env.close()