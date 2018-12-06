# Imports
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from SumTree import SumTree

import gym



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class ReplayMemory(object):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, size):
        self.size = size
        # self.buffer = []
        self.position = 0
        self.tree = SumTree(size)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def insert(self, transition, error):
        """ Saves a transition """
        p = self._get_priority(error)
        self.tree.add(p, transition)
        self.position = (self.position + 1) % self.size

    def can_sample(self, batch_size):
        return batch_size <= self.position

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        done_mask = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            # Define sample current boundaries
            a = segment * i
            b = segment * (i + 1)

            # Sample
            s = random.uniform(a, b)

            # Get the trajectory that corresponds to the sample
            (idx, p, data) = self.tree.get(s)
            if data == 0:
                print("s is: ", s)
                idx = idx-1
                p = self.tree.tree[idx]
                data = self.tree.data[idx - self.tree.capacity + 1]
            priorities.append(p)
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            done_mask.append(data[4])
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return states, actions, rewards, next_states, done_mask, idxs, is_weight


    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


    def __len__(self):
        return len(self.memory)


class CNN_NN_DQN(nn.Module):

    def unpack_arch(self, architecture):
        """
        :param architecture: dict. containing num. of layers in every layer (2-layer FC network)
        :return: all CNN architecture params
        """
        input_width = architecture["input_width"]
        input_height = architecture["input_height"]
        input_depth = architecture["input_depth"]
        action_space = architecture["action_space"]
        hidden_units_0 = architecture["hidden_units_0"]
        hidden_units_1 = architecture["hidden_units_1"]
        hidden_units_2 = architecture["hidden_units_2"]
        kernel = architecture["kernel"]
        stride = architecture["stride"]
        padding = architecture["padding"]
        pool = architecture["pool"]

        return input_width, input_height, input_depth, action_space, hidden_units_0, hidden_units_1, \
               hidden_units_2, kernel, stride, padding, pool

    def calc_flat_size(self, input_width, input_height, kernel, stride, padding, pool, last_depth):
        # No Padding
        out_1_width = (((input_width - kernel + 2*padding) / stride) + 1)/pool
        out_1_height = (((input_height - kernel + 2*padding) / stride) + 1)/pool
        out_2_width = (((out_1_width - kernel + 2*padding) / stride) + 1)/pool
        out_2_height = (((out_1_height - kernel + 2*padding) / stride) + 1)/pool
        out_3_width = (((out_2_width - kernel + 2*padding) / stride) + 1)/pool
        out_3_height = (((out_2_height - kernel + 2*padding) / stride) + 1)/pool
        flat_size = out_3_width * out_3_height * last_depth
        return flat_size

    def __init__(self, architecture):
        super(CNN_NN_DQN, self).__init__()

        input_width, input_height, input_depth, action_space, hidden_units_0, hidden_units_1,\
        hidden_units_2, kernel, stride, padding, pool = self.unpack_arch(architecture)

        flat_size = self.calc_flat_size(input_width, input_height, kernel, stride, padding, pool, hidden_units_2)

        self.conv1 = nn.Conv2d(input_depth, hidden_units_0, kernel_size=kernel, stride=stride, padding=padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=pool)
        self.bn1 = nn.BatchNorm2d(hidden_units_0)
        self.conv2 = nn.Conv2d(hidden_units_0, hidden_units_1, kernel_size=kernel, stride=stride, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool)
        self.bn2 = nn.BatchNorm2d(hidden_units_1)
        self.conv3 = nn.Conv2d(hidden_units_1, hidden_units_2, kernel_size=kernel, stride=stride, padding=padding)
        self.maxpool3 = nn.MaxPool2d(kernel_size=pool)
        self.bn3 = nn.BatchNorm2d(hidden_units_2)
        self.head = nn.Linear(int(flat_size), action_space)

    def forward(self, x):
        x = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.bn2(self.maxpool2(self.conv2(x))))
        x = F.relu(self.bn3(self.maxpool3(self.conv3(x))))
        return self.head(x.view(x.size(0), -1))

    def parameter_update(self, source):
        self.load_state_dict(source.state_dict())


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


# def plot_rewards(reward_list, avg_reward_list, num_iterations, save=True):
#
#     plt.plot(reward_list, color=(0, 0, 1, 0.3))
#     plt.plot(avg_reward_list, 'b')
#     plt.xlabel('# episodes')
#     plt.ylabel('# Acc. episodic reward')
#     plt.title('Accumulated episodic reward vs. num. of episodes, iterations=' + str(num_iterations+1))
#     if save:
#         plt.savefig('AccRewardVsEpisode_DQN')
#     plt.pause(0.05)
#     plt.clf()


def encode_states(states, num_states, encode_type):
    """
    Gets an array of integers and returns their one-hot encoding
    :param states: list of integers in [0,num_states-1]
    :param num_states: total number of states in env (500)
    :param encode_type: string - "one_hot" or "???"
    :return: states_one_hot: one hot encoding of states
    """
    if encode_type == "one_hot":
        batch_size = len(states)
        states_one_hot = np.zeros((batch_size, num_states))
        states_one_hot[np.arange(batch_size), states] = 1
        return states_one_hot

resize = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                         torchvision.transforms.Resize(40, interpolation=Image.CUBIC),
                                         torchvision.transforms.ToTensor()])
def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, :]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


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
        fig1.savefig('AccRewardVsEpisode_DQN_32units')
        fig2.savefig('AccRewardVsEpisode_DQN_32units_stdev')
        np.save('rewards_32units', reward_arr)
        np.save('avg_rewards_32units', avg_reward_arr)
        np.save('stdev_rewards_32units', stdev_reward_arr)

    fig1.clf()
    fig2.clf()
