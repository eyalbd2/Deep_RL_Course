
# Imports
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def insert(self, transition):
        """ Saves a transition """
        if len(self.buffer) < self.size:    # case buffer not full
            self.buffer.append(transition)
        else:                               # case buffer is full
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.size

    def can_sample(self, batch_size):
        return batch_size <= len(self.buffer)

    def sample(self, batch_size):
        transition_batch = random.sample(self.buffer, batch_size)
        states = [transition.state for transition in transition_batch]
        actions = np.array([transition.action for transition in transition_batch])
        rewards = np.array([transition.reward for transition in transition_batch])
        next_states = [transition.next_state for transition in transition_batch]
        done_mask = np.array([transition.done for transition in transition_batch])
        return states, actions, rewards, next_states, done_mask

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def unpack_arch(self, architecture):
        """
        :param architecture: dict. containing num. of layers in every layer (2-layer FC network)
        :return: input_size, hidden_size, output_size of NN
        """

        input_size = architecture["num_states"]
        hidden_size = architecture["hidden_units"]
        output_size = architecture["num_actions"]
        return input_size, hidden_size, output_size

    def __init__(self, architecture):
        super(DQN, self).__init__()

        num_states, hidden_units, num_actions = self.unpack_arch(architecture)

        self.hidden = nn.Linear(num_states, hidden_units)
        self.out = nn.Linear(hidden_units, num_actions)


    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.out(x)


def plot_rewards(reward_list, avg_reward_list, num_iterations, save=True):

    plt.plot(reward_list, color=(0, 0, 1, 0.3))
    plt.plot(avg_reward_list, 'b')

    plt.xlabel('# episodes')
    plt.ylabel('# Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes, iterations=' + str(num_iterations+1))
    if save:
        plt.savefig('AccRewardVsEpisode_DQN')
    plt.pause(0.05)
    plt.clf()

