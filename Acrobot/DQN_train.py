
import gym
import torch.optim as optim
from deep_q_learner import deep_Q_learning
import time
from utils import OptimizerSpec

env = gym.make('Acrobot-v1')

# Define all the model parameters
replay_buffer_size = 100000     # buffer size
start_learning = 50000          # num. transitions before start learning
target_update_freq = 500        # num. transitions between Q_target network updates
eps = 0.03                      # final epsilon for epsilon-greedy action selection
schedule_timesteps = 250000     # num. transitions for epsilon annealing
batch_size = 32                 # size of batch size for training
gamma = 0.99                    # discount factor of MDP
eps_optim = 0.01                # epsilon parameter for optimization (improves stability of optimizer)
alpha = 0.95                    # alpha parameter of RMSprop optimizer
learning_rate = 0.00025         # step size for optimization process

save_fig = True     # whether to save figure of accumulated reward
save_model = True   # whether to save the DQN model

exploration_params = {"timesteps": schedule_timesteps, "final_eps": eps}

optimizer_spec = OptimizerSpec(constructor=optim.RMSprop,
                               kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps_optim),)

start_time = time.time()
deep_Q_learning(env, optimizer_spec, exploration_params, replay_buffer_size, start_learning,
                batch_size, gamma, target_update_freq, save_fig, save_model)
end_time = time.time()

print('Tot. running time: ', end_time-start_time)
