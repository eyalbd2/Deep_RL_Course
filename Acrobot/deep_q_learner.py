
# Imports
import torch
from itertools import count
from torch.autograd import Variable
from utils import *
import random
import numpy as np


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deep_Q_learning(env, optimizer_spec, exploration_params, replay_buffer_size=100000,
                    start_learning=50000, batch_size=128, gamma=0.99, target_update_freq=10000,
                    save_fig=True, save_model=False):
    """
        Implementation of DQN learning procedure
    :param env: gym environment
    :param architecture: dict. with input_size, hidden_size and output_size (2-layer NN)
    :param optimizer_spec: optimizer and its params
    :param encode_type: how to encode state - one_hot or ???
    :param exploration_params: dict. with final epsilon and num. of time steps until final epsilon
    :param replay_buffer_size: size of replay memory
    :param start_learning: num. iterations before start learning (filling the buffer)
    :param batch_size: batch size for optimization steps
    :param gamma: discount factor of MDP
    :param target_update_freq: num. of iterations between target network update
    :param save_fig: flag for saving plots
    :param save_model: flag for saving optimal weirhts of the net at the end of training session
        Algorithm saves a trained network
    """

    def select_epsilon_greedy_action(model, state, exploration_params, t):
        """
        :param model: Q network
        :param state: current state of env - in 3D image difference
        :param exploration_params: final epsilon and num. timesteps until final epsilon
        :param t: current timestep
        :return: Algorithm returns an action chosen by an epsilon greedy policy
        """
        # Compute current epsilon
        fraction = min(1.0, float(t) /exploration_params["timesteps"])
        epsilon = 1 + fraction * (exploration_params["final_eps"] - 1)

        num_actions = model.head.out_features    # output size of Q network is as action space

        sample = random.random()
        if sample <= epsilon:
            return random.randrange(num_actions), epsilon
        else:
            return int(model(Variable(state)).data.argmax()), epsilon

    num_actions = env.action_space.n

    # Initialize network and target network
    Q = DQN(num_actions).to(device)
    Q_target = DQN(num_actions).to(device)

    # Construct optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = PriorReplayMemory(replay_buffer_size)

    # Initialize episodic reward list
    episodic_rewards = []
    avg_episodic_rewards = []
    stdev_episodic_rewards = []
    best_avg_episodic_reward = -np.inf
    acc_episodic_reward = 0.0

    num_param_updates = 0
    episodes_passed = 0
    stopping_counter = 0

    _ = env.reset()
    current_screen = get_screen(env)
    state = current_screen

    for t in count():
        # Stop if last average accumulated episodic reward over 10 episodes is above -150
        if len(avg_episodic_rewards) > 0:
            if avg_episodic_rewards[-1] > -115:
                stopping_counter += 1
                if stopping_counter >= 11:
                    if save_model:
                        torch.save(Q, 'stable_trained_Acrobot_model_v4')
                    break
            else:
                stopping_counter = 0

        # Choose random action if not yet start learning
        if t > start_learning:
            action, eps_val = select_epsilon_greedy_action(Q, state, exploration_params, t)
        else:
            action = random.randrange(num_actions)
            eps_val = 1.0

        # Advance one step
        _, reward, done, _ = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen

        # Construct priority for the current sample
        # Q value for state-action pair that were taken
        current_Q_value = Q(state)[0][action]
        # Best Q value from next state - using Q_target as estimator
        next_Q_value = Q_target(next_state).detach().max(1)[0]
        # Compute estimated Q values (based on Q_target)
        target_Q_value = reward + (gamma * next_Q_value)
        # Compute Bellman error
        bellman_error = target_Q_value - current_Q_value.squeeze()

        # document accumulated reward
        acc_episodic_reward = acc_episodic_reward + reward
        # Save and insert transition to replay buffer
        transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=int(done))
        replay_buffer.insert(transition, np.abs(bellman_error.data))
        # Resets the environment when reaching an episode boundary.

        if done:
            # Resets the environment when finishing an episode
            _ = env.reset()
            current_screen = get_screen(env)
            next_state = current_screen

            # Document statistics
            episodic_rewards.append(acc_episodic_reward)
            acc_episodic_reward = 0.0
            episodes_passed += 1

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
                    torch.save(Q, 'trained_DQN_model')

            # Update plot of acc. rewards every 20 episodes and print
            # training details
            if episodes_passed % 20 == 0:
                plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                             np.array(stdev_episodic_rewards), save_fig)
                print('Episode {}\tAvg. Reward: {:.2f}\tEpsilon: {:.4f}\t'.format(
                    episodes_passed, avg_episodic_rewards[-1], eps_val))
                print('Best avg. episodic reward:', best_avg_episodic_reward)

        state = next_state

        # Perform experience replay and train the network.
        if t > start_learning and replay_buffer.can_sample(batch_size):
            # Sample from experience buffer
            state_batch, action_batch, reward_batch, next_state_batch, done_mask, idxs_batch, is_weight = \
                replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            state_batch = torch.cat(state_batch)
            action_batch = Variable(torch.tensor(action_batch).long())
            reward_batch = Variable(torch.tensor(reward_batch, device=device)).type(dtype)
            next_state_batch = torch.cat(next_state_batch)
            not_done_mask = Variable(1 - torch.tensor(done_mask)).type(dtype)
            is_weight = Variable(torch.tensor(is_weight)).type(dtype)

            # Case GPU is available
            if USE_CUDA:
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()

            # Q values for state-action pair that were taken
            current_Q_values = Q(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
            # Best Q values from next state - using Q_target as estimator
            Q_max_next_state = Q_target(next_state_batch).detach().max(1)[0]
            # Update only when episode not terminated
            next_Q_values = not_done_mask * Q_max_next_state
            # Compute estimated Q values (based on Q_target)
            target_Q_values = reward_batch + (gamma * next_Q_values)
            # Compute TD error
            loss = (current_Q_values - target_Q_values.detach()).pow(2) * is_weight
            prios = loss + 1e-5
            loss = loss.mean()
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # Run backward pass
            loss.backward()
            # update priority
            for i in range(batch_size):
                idx = idxs_batch[i]
                replay_buffer.update(idx, prios[i].data.cpu().numpy())
            # Perform the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to Q_target network
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())
