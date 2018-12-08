
# Imports
import gym
import numpy as np
from itertools import count
from utils import AC_model, plot_rewards

import torch
import torch.optim as optim


env = gym.make('Taxi-v2')

print_every = 100
hidden_units = 32
gamma = 1
save_model = False
save_fig = False

# Define architecture parameters
architecture = {"state_dim": env.observation_space.n,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}

# Initialize AC model
AC_net = AC_model(architecture)
# Define optimizer
optimizer = optim.Adam(AC_net.parameters(), lr=1e-3)

episodes_passed = 0
acc_rewards = []
last_t = 0
state = env.reset()

# Initialize episodic reward list
episodic_rewards = []
avg_episodic_rewards = []
stdev_episodic_rewards = []
acc_episodic_reward = 0.0
best_avg_episodic_reward = -np.inf


for t in count():

    if len(avg_episodic_rewards) > 0:   # so that avg_episodic_rewards won't be empty
        # Stop if max episodes or playing good (above avg. reward of 5 over last 10 episodes)
        # if episodes_passed == 5000 or avg_episodic_rewards[-1] > 5:
        if episodes_passed == 20000:
            break

    action = AC_net.select_action(state)   # Take action
    state, reward, done, _ = env.step(action)   # Get transition
    AC_net.rewards.append(reward)               # Document reward
    acc_episodic_reward = acc_episodic_reward + reward  # Document accumulated episodic reward

    # Episode ends - reset environment and document statistics
    if reward == 20:
    # if done:
        episodes_passed += 1
        episodic_rewards.append(acc_episodic_reward)
        acc_episodic_reward = 0.0

        # Compute average reward and variance (standard deviation)
        if len(episodic_rewards) <= 10:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
            if len(episodic_rewards) >= 2:
                stdev_episodic_rewards.append(np.std(np.array(episodic_rewards)))

        else:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))
            stdev_episodic_rewards.append(np.std(np.array(episodic_rewards[-10:])))

        # Check if average acc. reward has improved
        if avg_episodic_rewards[-1] > best_avg_episodic_reward:
            best_avg_episodic_reward = avg_episodic_rewards[-1]
            if save_model:
                torch.save(AC_net, 'trained_AC_model')

        # Update plot of acc. rewards every 20 episodes and print
        # training details
        if episodes_passed % print_every == 0:
            plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                         np.array(stdev_episodic_rewards), save_fig)
            print('Episode {}\tLast episode length: {:5d}\tAvg. Reward: {:.2f}\t'.format(
                episodes_passed, t - last_t, avg_episodic_rewards[-1]))
            print('Best avg. episodic reward:', best_avg_episodic_reward)

        last_t = t  # Follow episodes length
        state = env.reset()
        AC_net.update_weights(optimizer)    # Perform network weights update
        continue
