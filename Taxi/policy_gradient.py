
from itertools import count
import torch
from torch.autograd import Variable
from utils import *
from torch.distributions import Categorical
import pandas as pd

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


pg_OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


def policy_gradient(env, architecture, optimizer_spec, learning_rate, encode_type,
                    num_of_episodes, gamma=0.99, save_fig=True):
    """
            Implementation of Policy Gradient learning procedure using PPO

        :param env: gym environment
        :param architecture: dict. with input_size, hidden_size and output_size (2-layer NN)
        :param optimizer_spec: optimizer and its params
        :param learning_rate: learning rate
        :param encode_type: how to encode state - always one_hot in PG
        :param num_of_episodes: how many episodes to run during training
        :param gamma: discount factor of MDP
        :param target_update_freq: num. of iterations between target network update

            Algorithm saves a trained network
    """

    def select_action(model, state):
        """
        :param model: policy network
        :param state: current state of env
        :return: Algorithm returns an action chosen by policy probabilities and log of the probabilities
        """
        num_states = list(model.children())[0].in_features
        state = encode_states([state], num_states, encode_type)
        action_probs = model(Variable(torch.from_numpy(state).type(dtype)))
        c = Categorical(action_probs)
        action = c.sample()
        log_probs = c.log_prob(action)

        return action, log_probs

    def update_policy(reward_episode, policy_history, loss_history, reward_history):
        discounted_reward = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reward_episode[::-1]:
            discounted_reward = r + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        if rewards.std() is not None:
            rewards = (rewards - rewards.mean()) / (rewards.std())
            # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        else:
            rewards = (rewards - rewards.mean())

        # Calculate loss
        loss = (torch.sum(torch.mul(policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save and intialize episode history counters
        loss_history.append(loss.item())
        reward_history.append(np.sum(reward_episode))
        policy_history = Variable(torch.Tensor())
        reward_episode = []

        return loss_history, reward_history, policy_history, reward_episode

    policy = Policy(architecture)
    optimizer = optimizer_spec.constructor(policy.parameters(), **optimizer_spec.kwargs)
    running_reward = 10
    reward_episode = []
    policy_history = Variable(torch.Tensor())
    loss_history = []
    reward_history = []

    for episode in range(num_of_episodes):
        state = env.reset()  # Reset environment and get the starting state

        for time in range(1000):
            taken_action, action_log_prob = select_action(policy, state)
            if policy_history.dim() != 0:
                policy_history = torch.cat([policy_history, action_log_prob])
            else:
                policy_history = (action_log_prob)

            # Step through environment using chosen action
            state, reward, done, _ = env.step(int(taken_action.data[0]))
            # Save reward
            reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        loss_history, reward_history, policy_history, reward_episode = \
            update_policy(reward_episode, policy_history, loss_history, reward_history)

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
        #     break

    window = int(num_of_episodes / 20)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9]);
    rolling_mean = pd.Series(reward_history).rolling(window).mean()
    std = pd.Series(reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(reward_history)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode');
    ax1.set_ylabel('Episode Length')

    ax2.plot(reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode');
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()
    # fig.savefig('results.png')