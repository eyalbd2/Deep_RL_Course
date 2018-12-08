
import torch
import gym
import numpy as np

# Import environment
env = gym.make('Taxi-v2')
# Define model and load pre-trained weights
trained_AC_model = torch.load('trained_AC_model')

# Reset env
state = env.reset()
# Present trained behaviour over episodes
num_test_episodes = 10

episodes_passed = 0
acc_episodic_reward = 0.0

while episodes_passed < num_test_episodes:
    # Choose action greedily
    action = trained_AC_model.select_action(state)
    # Act on env
    state, reward, done, _ = env.step(action)
    # Add to accumulative reward
    acc_episodic_reward += reward
    # When episode is done - reset and print (limit to 200 transitions in test)
    if done:
        # Print acc. reward
        print('Episode {}\tAccumulated Reward: {:.2f}\t'.format(
            episodes_passed+1, acc_episodic_reward))
        # Update statistics
        episodes_passed += 1
        acc_episodic_reward = 0.0

        state = env.reset()






