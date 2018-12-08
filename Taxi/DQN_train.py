
# Imports
import gym
import torch.optim as optim
from utils import OptimizerSpec
import time
from deep_q_learner import deep_Q_learning


# Import Taxi environment
env = gym.make('Taxi-v2')

# Define all the model parameters
hidden_units = 32               # num. units in hidden layer
replay_buffer_size = 200000     # buffer size
start_learning = 50000          # num. transitions before start learning
target_update_freq = 10000      # num. transitions between Q_target network updates
eps = 0.1                       # final epsilon for epsilon-greedy action selection
schedule_timesteps = 350000     # num. transitions for epsilon annealing
batch_size = 32                 # size of batch size for training
gamma = 0.99                    # discount factor of MDP
eps_optim = 0.01                # epsilon parameter for optimization (improves stability of optimizer)
alpha = 0.95                    # alpha parameter of RMSprop optimizer
learning_rate = 0.00025         # step size for optimization process


encode_method = 'one_hot'     # state encoding method ('one_hot' or 'positions')

if encode_method is 'positions':
    state_dim = 19  # explained in utils.encode_states
else:   # one-hot
    state_dim = env.observation_space.n

regularization = None           # regularization may be 'regularization'
save_fig = False        # whether to save figure of accumulated reward
save_model = False      # whether to save the DQN model

# Define 2-layered architecture
architecture = {"state_dim": state_dim,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}
# Pack the epsilon greedy exploration parameters
exploration_params = {"timesteps": schedule_timesteps, "final_eps": eps}
# Define optimization procedure
optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps_optim),)
        # kwargs=dict(lr=learning_rate),)


t_start = time.time()
# Run deep Q-learning algorithm
deep_Q_learning(env, architecture, optimizer_spec,
                exploration_params, replay_buffer_size, start_learning,
                batch_size, gamma, target_update_freq, regularization,
                encode_method, save_fig, save_model)

t_end = time.time()
print('Total learning time: ', t_end-t_start)
