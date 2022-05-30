import numpy as np
from EnvKinova_gym import EnvKinova_gym
import utilities as U
import time
from graphics import *

env = EnvKinova_gym()
print(env.observation_space)
n_observations = 100 ** 2 # env.n
n_actions = 9 #np.prod(env.action_space.shape)

# Index actions 

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
print("QTABLE", Q_table)

# CONSTANTS
N_EPISODES = 10000
MAX_ITER_EPISODE = 100
EXPLORATION_PROB = 1
DECAY = 0.001
MIN_EXPLORATION_PROB = 0.01
GAMMA = 0.99
LR = 0.1

total_rewards_episode = []
gTables = Graphics()

#we iterate over episodes
for e in range(N_EPISODES):
    #we initialize the first state of the episode
    current_state = env.reset()
    current_state = U.state2index(current_state)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    done = False
    info = {}
    
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(MAX_ITER_EPISODE): 
        # we sample a float from a uniform distribution over 0 and 1
        # if the sampled flaot is less than the exploration proba
        #     the agent selects arandom action
        # else
        #     he exploits his knowledge using the bellman equation 
        
        if np.random.uniform(0,1) < EXPLORATION_PROB:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
            x = action // 3
            y = action % 3
            action = x-1, y-1
            # print("OWUUWOUWOWUOWUWOUJWOUWOUWOWUWOUWO")
            # time.sleep(0.5)
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, info = env.step(action)
        action = U.action2index(action)
        next_state = U.state2index(next_state)
        
        print("EPISODE", e, "ITER", i, "STATE", current_state, "ACTION", action)

        # We update our Q-table using the Q-learning iteration
        Q_table[current_state, action] = (1 - LR) * Q_table[current_state, action] + LR * (reward + GAMMA * max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
        # print(Q_table)
        # time.sleep(1)

    gTables.storeData(EXPLORATION_PROB, info["arrival"], info["far"], info["dist"])
    
    register, show = 10, 500
    if e != 0 and e % register == 0:
        gTables.insertData()

    if e != 0 and e % show == 0:
        gTables.show(e, register)

    #We update the exploration proba using exponential decay formula 
    EXPLORATION_PROB = max(MIN_EXPLORATION_PROB, np.exp(-DECAY*e))
    total_rewards_episode.append(total_episode_reward)

    print("Mean reward per thousand episodes")
    for i in range(10):
        print((i+1)*1000,": mean espiode reward: ", np.mean(total_rewards_episode[1000*i:1000*(i+1)]))

