
# Imports
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
import matplotlib.pyplot as plt
from torch.distributions import Categorical

'''
    This script includes the implementations of all the necessary/auxiliary Classes/Functions:
    - ReplayMemory - Replay buffer
    - DQN - deep Q network 
    - AC_model - deep network for actor critic learning
    - unpack_arch - unpacking input architecture dictionary 
    - plot_rewards - plotting accumulated rewards  
    - encode_states - encoding of states from an integer
    - decode_positions - decodes taxi's, passenger and destinations positions from integer state
'''

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])  # For AC model


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

    def __init__(self, architecture, is_dropout=False):
        super(DQN, self).__init__()

        state_dim, hidden_units, num_actions = unpack_arch(architecture)
        self.state_dim = state_dim      # 500 or 19 depends on encoding
        self.is_dropout = is_dropout    # should use dropout regularization

        self.hidden = nn.Linear(state_dim, hidden_units)
        self.out = nn.Linear(hidden_units, num_actions)
        # dropout layers
        if is_dropout:
            self.do1 = nn.Dropout(p=0.2)
            self.do2 = nn.Dropout(p=0.2)



    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        if self.is_dropout:
            x = self.do1(x)

        x = self.out(x)
        if self.is_dropout:
            x = self.do2(x)

        return x

    def select_action(self, state):
        # Find out encoding method based on input size
        if self.state_dim == 19:    # explained in utils.encode_states
            encode_method = 'positions'
        else:  # input size is 500
            encode_method = 'one-hot'
        # Selects action greedily
        encoded_state = encode_states([state], encode_method, self.state_dim)
        encoded_state = Variable(torch.from_numpy(encoded_state).type(torch.FloatTensor))
        Q_vals = self.forward(encoded_state)
        return int(Q_vals.data.argmax())


class AC_model(nn.Module):
    def __init__(self, architecture):
        super(AC_model, self).__init__()

        num_states, hidden_units, num_actions = unpack_arch(architecture)

        # Define shared network - action head and value head
        self.hidden = nn.Linear(num_states, hidden_units)
        self.action_head = nn.Linear(hidden_units, num_actions)
        self.value_head = nn.Linear(hidden_units, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state):
        """
            This function selects action which is sampled from the
            action head of the network (policy) - a~pi(:,s)

        :param state: state of the environment
        :return: action
        """

        # One hot encoding
        state_one_hot = np.zeros((1, self.hidden.in_features))
        state_one_hot[0, int(state)] = 1

        state = torch.from_numpy(state_one_hot).float()
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def update_weights(self, optimizer):
        """
            This function applies optimization step to the network based on a trajectory
            where the loss os the sum of minus the expected return and the squared TD
            error for the value function, V
        """

        gamma = 1   # Finite horizon
        eps = np.finfo(np.float32).eps.item()  # For stabilization

        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)    # Normalize - stabilize

        # Calculate losses
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value.squeeze(dim=1), torch.tensor([r])))

        # Apply optimization step
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()
        # Empty history of last trajectory
        del self.rewards[:]
        del self.saved_actions[:]


def unpack_arch(architecture):
    """
    :param architecture: dict. containing num. of units in every layer (2-layer FC network)
    :return: input_size, hidden_size, output_size of NN
    """

    input_size = architecture["state_dim"]
    hidden_size = architecture["hidden_units"]
    output_size = architecture["num_actions"]
    return input_size, hidden_size, output_size


def plot_rewards(reward_arr, avg_reward_arr, stdev_reward_arr, save=True):

    fig1 = plt.figure(1)
    # rewards + average rewards
    plt.plot(reward_arr, color='b', alpha=0.3)
    plt.plot(avg_reward_arr, color='b')
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Acc. episodic reward', 'Avg. acc. episodic reward'])
    plt.tight_layout()

    # average rewards + stdevs
    fig2 = plt.figure(2)
    plt.plot(avg_reward_arr, color='b')
    plt.fill_between(range(1, len(avg_reward_arr)), avg_reward_arr[1:] - stdev_reward_arr,
                     avg_reward_arr[1:] + stdev_reward_arr, color='b', alpha=0.2)
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Avg. acc. episodic reward', 'Stdev envelope of acc. episodic reward'])
    plt.tight_layout()

    plt.pause(0.01)

    if save:
        fig1.savefig('AccRewardVsEpisode_AC_finite')
        fig2.savefig('AccRewardVsEpisode_AC_stdev_finite')
        np.save('rewards_AC_finite', reward_arr)
        np.save('avg_rewards_AC_finite', avg_reward_arr)
        np.save('stdev_rewards_AC_finite', stdev_reward_arr)

    fig1.clf()
    fig2.clf()


def encode_states(states, encode_method, state_dim):
    """
        Gets a list of integers and returns their encoding
        as 1 of 2 possible encoding methods:
            - one-hot encoding (array)
            - position encoding

    :param states: list of integers in [0,num_states-1]
    :param encode_method: one of 'one_hot', 'positions'
    :param state_dim: dimension of state (used for 'one_hot' encoding)
    :return: states_encoded: one hot encoding of states
    """

    batch_size = len(states)

    if encode_method is 'positions':
        '''
            position encoding encodes the important game positions as 
            a 19-dimensional vector:
                - 5 dimensions are used for one-hot encoding of the taxi's row (0-4)
                - 5 dimensions are used for one-hot encoding of the taxi's col (0-4)
                - 5 dimensions are used for one-hot encoding of the passenger's position:
                    0 is 'R', 1 is 'G', 2 is 'Y', 3 is 'B' and 4 is if the passenger in the taxi
                - 4 dimensions are used for one-hot encoding of the destination location:
                    0 is 'R', 1 is 'G', 2 is 'Y' and 3 is 'B'
                we simply concatenate those vectors into a 19-dim. vector with 4 ones in it
                corresponding to the positions encoding and the rest are zeros.      
        '''

        taxi_row, taxi_col, pass_code, dest_loc = decode_positions(states)

        # one-hot encode taxi's row
        taxi_row_onehot = np.zeros((batch_size, 5))
        taxi_row_onehot[np.arange(batch_size), taxi_row] = 1
        # one-hot encode taxi's col
        taxi_col_onehot = np.zeros((batch_size, 5))
        taxi_col_onehot[np.arange(batch_size), taxi_col] = 1
        # one-hot encode row
        pass_code_onehot = np.zeros((batch_size, 5))
        pass_code_onehot[np.arange(batch_size), pass_code] = 1
        # one-hot encode row
        dest_loc_onehot = np.zeros((batch_size, 4))
        dest_loc_onehot[np.arange(batch_size), dest_loc] = 1

        states_encoded = np.concatenate([taxi_row_onehot, taxi_col_onehot,
                                         pass_code_onehot, dest_loc_onehot], axis=1)

    else:   # one-hot
        states_encoded = np.zeros((batch_size, state_dim))
        states_encoded[np.arange(batch_size), states] = 1

    return states_encoded


def decode_positions(states):
    """
    Gets an state from env.render() (int) and returns
    the taxi position (row, col), the passenger position
    and the destination location

    :param states: a list of states represented as integers [0-499]
    :return: taxi_row, taxi_col, pass_code, dest_idx
    """
    dest_loc = [state % 4 for state in states]
    states = [state // 4 for state in states]
    pass_code = [state % 5 for state in states]
    states = [state // 5 for state in states]
    taxi_col = [state % 5 for state in states]
    states = [state // 5 for state in states]
    taxi_row = states
    return taxi_row, taxi_col, pass_code, dest_loc

