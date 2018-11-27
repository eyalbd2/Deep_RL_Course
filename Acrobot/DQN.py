

from itertools import count
from torch.autograd import Variable
from utills import *


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def deep_Q_learning(env, architecture, optimizer_spec, exploration_params, replay_buffer_size=100000,
                    start_learning=50000, batch_size=128, gamma=0.99, target_update_freq=10000, save_fig=True):
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
            return random.randrange(num_actions)
        else:
            return int(model(Variable(state)).data.argmax())


    num_actions = env.action_space.n

    # Initialize network and target network
    Q = CNN_NN_DQN(architecture)
    Q_target = CNN_NN_DQN(architecture)

    # Construct optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayMemory(replay_buffer_size)

    # Initialize episodic reward list
    episodic_rewards = []
    avg_episodic_rewards = []
    acc_episodic_reward = 0.0

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    _ = env.reset()
    current_screen = get_screen(env)
    state = current_screen
    ## DEBUG ##
    # last_t = 0
    ## ##### ##
    episodes_passed = 0

    for t in count():
        # TODO: may add stopping criterion

        # Choose random action if not yet start learning
        if t > start_learning:
            action = select_epsilon_greedy_action(Q, state, exploration_params, t - start_learning)
        else:
            action = random.randrange(num_actions)

        # Advance one step
        _, reward, done, _ = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen
        # document accumulated reward
        acc_episodic_reward = acc_episodic_reward + reward
        # Save and insert transition to replay buffer
        transition = Transition(state=state, action=action, reward=reward,
                                next_state=next_state, done=int(done))
        replay_buffer.insert(transition)
        # Resets the environment when reaching an episode boundary.
        if done:
            env.reset()
            current_screen = get_screen(env)
            next_state = current_screen

            ## DEBUG ##
            # episode_len = t - last_t
            # last_t = t
            # if t % 1000:
            # print(acc_episodic_reward, episode_len)

            episodic_rewards.append(acc_episodic_reward)
            acc_episodic_reward = 0.0
            episodes_passed += 1

            # Compute average reward
            if len(episodic_rewards) <= 10:
                avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
            else:
                avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))

            # Plot result every 100 episodes
            if episodes_passed % 20 == 0:
                plot_rewards(episodic_rewards, avg_episodic_rewards, t, save_fig)

        state = next_state

        # Perform experience replay and train the network.
        if t > start_learning and replay_buffer.can_sample(batch_size):
            # Sample from experience buffer
            state_batch, action_batch, reward_batch, next_state_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            state_batch = torch.cat(state_batch)
            action_batch = Variable(torch.from_numpy(action_batch).long())
            reward_batch = Variable(torch.from_numpy(reward_batch)).type(dtype)
            next_state_batch = torch.cat(next_state_batch)
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
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values.squeeze()
            # Clip the bellman error between [-1 , 1] - gradient explosion
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient (inner derivative)
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # Run backward pass
            current_Q_values.backward(d_error.data.unsqueeze(1))

            # Perform the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to Q_target network
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())


            # Document statistics
            # episode_rewards = []
            #
            # if len(episode_rewards) > 0:
            #     mean_episode_reward = np.mean(episode_rewards[-100:])
            # if len(episode_rewards) > 100:
            #     best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)