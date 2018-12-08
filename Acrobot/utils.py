
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class PriorReplayMemory(object):
    # Prioritized ER params
    e = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, size):
        self.size = size
        # self.buffer = []
        self.position = 0
        self.tree = SumTree(size)

    def _get_priority(self, error):
        return (error + self.e) ** self.alpha

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


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(64)

        self.head = nn.Linear(1600, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.bn2(self.maxpool2(self.conv2(x))))
        x = F.relu(self.bn3(self.maxpool3(self.conv3(x))))
        return self.head(x.view(x.size(0), -1))

    def parameter_update(self, source):
        self.load_state_dict(source.state_dict())

    def select_action(self, state):
        Q_vals = self.forward(state)
        return int(Q_vals.data.argmax())


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


# Resize operation - to 40 x 40 x 3
resize = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                         torchvision.transforms.Resize(40, interpolation=Image.CUBIC),
                                         torchvision.transforms.ToTensor()])


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
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
