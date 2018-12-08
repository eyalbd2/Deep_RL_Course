
# Imports
import torch
from itertools import count
from torch.autograd import Variable
import numpy as np
from utils import *


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def deep_Q_learning(env, architecture, optimizer_spec,
                    exploration_params, replay_buffer_size=100000, start_learning=50000,
                    batch_size=128, gamma=0.99, target_update_freq=10000, regularization=None,
                    encode_method='one_hot', save_fig=True, save_model=True):

    """
        Implementation of DQN learning procedure
    
    :param env: gym environment
    :param architecture: dict. with input_size, hidden_size and output_size (2-layer NN)
    :param optimizer_spec: optimizer and its params
    :param exploration_params: dict. with final epsilon and num. of time steps until final epsilon
    :param replay_buffer_size: size of replay memory
    :param start_learning: num. iterations before start learning (filling the buffer)
    :param batch_size: batch size for optimization steps
    :param gamma: discount factor of MDP
    :param target_update_freq: num. of iterations between target network update
    :param save_fig: whether to save figure of reward vs. # episode

        Algorithm saves a trained network
    """

    def select_epsilon_greedy_action(model, state, exploration_params, t):
        """
            Function returns an action using epsilon greedy with linear
            annealing schedule (starting from 1 and annealing linearly
            till exploration_params["final_eps"])


        :param model: Q network
        :param state: current state of env
        :param exploration_params: final epsilon and num. timesteps until final epsilon
        :param t: current time-step
        :return: an action chosen by an epsilon greedy policy
        """
        # Compute current epsilon
        fraction = min(1.0, float(t)/exploration_params["timesteps"])
        epsilon = 1 + fraction*(exploration_params["final_eps"] - 1)

        num_actions = model.out.out_features    # output size of Q network is as action space
        state_dim = list(model.children())[0].in_features

        sample = random.random()
        if sample <= epsilon:
            return random.randrange(num_actions), epsilon
        else:
            state = encode_states([state], encode_method, state_dim)
            state = Variable(torch.from_numpy(state).type(dtype))
            return int(model(state).data.argmax()), epsilon

    num_actions = env.action_space.n        # action space (=6)
    state_dim = architecture['state_dim']   # state space (=depends on encoding - 500/19)

    # Initialize network and target network
    is_dropout = False
    if regularization is 'dropout':
        is_dropout = True
    Q = DQN(architecture, is_dropout)
    Q_target = DQN(architecture, is_dropout)

    # Construct optimizer over the networks weights
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayMemory(replay_buffer_size)

    # Initialize episodic reward list
    episodic_rewards = []
    avg_episodic_rewards = []
    stdev_episodic_rewards = []
    acc_episodic_reward = 0.0
    best_avg_episodic_reward = -np.inf

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    state = env.reset()
    episodes_passed = 0
    last_t = 0  # to track episodes length

    for t in count():
        # Stopping criterion - max num. episodes
        if episodes_passed > 3000:
            break

        # Choose random action if not yet start learning
        if t > start_learning:
            action, eps_val = select_epsilon_greedy_action(Q, state, exploration_params, t)
        else:
            action = random.randrange(num_actions)
            eps_val = 1

        # s_enc = encode_states([state], encode_method, state_dim)
        # if s_enc[0][14] == 1:
        #     env.render()

        # Take action over env.
        next_state, reward, done, _ = env.step(action)
        # Document accumulated reward
        acc_episodic_reward = acc_episodic_reward + reward
        # Save and insert transition to replay buffer
        transition = Transition(state=state, action=action, reward=reward,
                                next_state=next_state, done=int(done))
        replay_buffer.insert(transition)

        if done:
            # Resets the environment when finishing an episode
            next_state = env.reset()
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
                print('Episode {}\tLast episode length: {:5d}\tAvg. Reward: {:.2f}\tEpsilon: {:.4f}\t'.format(
                    episodes_passed, t - last_t, avg_episodic_rewards[-1], eps_val))
                print('Best avg. episodic reward:', best_avg_episodic_reward)

            last_t = t

        state = next_state

        # Train network by sampling from experience buffer
        if t > start_learning and replay_buffer.can_sample(batch_size):
            # Sample from experience buffer
            state_batch, action_batch, reward_batch, next_state_batch, done_mask = replay_buffer.sample(batch_size)
            # One-hot encoding of states
            state_batch = encode_states(state_batch, encode_method, state_dim)
            next_state_batch = encode_states(next_state_batch, encode_method, state_dim)
            # Convert numpy nd_array to torch Variables for calculation
            state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
            action_batch = Variable(torch.from_numpy(action_batch).long())
            reward_batch = Variable(torch.from_numpy(reward_batch)).type(dtype)
            next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            # Case GPU is available
            if USE_CUDA:
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()

            # Q values for state-action pair that were taken
            current_Q_values = Q(state_batch).gather(1, action_batch.unsqueeze(1))
            # Best Q values from next state - using Q_target as estimator
            Q_max_next_state = Q_target(next_state_batch).detach().max(1)[0]
            # Update only when episode not terminated
            next_Q_values = not_done_mask * Q_max_next_state
            # Compute estimated Q values (based on Q_target)
            target_Q_values = reward_batch + (gamma * next_Q_values)

            # Compute MSE loss (of bellman error)
            loss = F.mse_loss(current_Q_values, target_Q_values.unsqueeze(1))
            # Case regularization
            if regularization is 'l1':
                l1_regularization = torch.tensor(0).type(dtype)
                for param in Q.parameters():
                    l1_regularization += torch.norm(param, 1)
                    # add l1 regularization to loss
                loss += 0.05*l1_regularization

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # Clip the bellman error between [-1 , 1] - prevent gradient explosion
            for param in Q.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()    # perform the update


            # # Compute Bellman error
            # bellman_error = target_Q_values - current_Q_values.squeeze()
            # # Clip the bellman error between [-1 , 1] - prevent gradient explosion
            # clipped_bellman_error = bellman_error.clamp(-1, 1)
            # # Note: clipped_bellman_delta * -1 will be right gradient
            # d_error = clipped_bellman_error * -1.0
            # # Clear previous gradients before backward pass
            # optimizer.zero_grad()
            # # Run backward pass
            # current_Q_values.backward(d_error.data.unsqueeze(1))
            # optimizer.step()    # perform the update

            num_param_updates += 1

            # Periodically update the target network by copying Q network into Q target
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())


