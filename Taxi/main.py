
# Imports
import gym
import torch.optim as optim
from deep_q_learner import dqn_OptimizerSpec, deep_Q_learning
from policy_gradient import pg_OptimizerSpec, policy_gradient


env = gym.make('Taxi-v2').unwrapped

dqn_hidden_units = 100
pg_hiden_units = 128
replay_buffer_size = 1000000
start_learning = 50000
learning_rate = 0.00025
alpha = 0.95
eps = 0.01
batch_size = 128
gamma = 0.99
target_update_freq = 10000
schedule_timesteps = 1000000
final_eps = 0.1
num_of_episodes = 100000
save_fig = True
encode_type = "one_hot"

dqn_architecture = {"num_states": env.observation_space.n, "hidden_units": dqn_hidden_units, "num_actions": env.action_space.n}
pg_architecture = {"num_states": env.observation_space.n, "hidden_units": pg_hiden_units, "num_actions": env.action_space.n}
exploration_params = {"timesteps": schedule_timesteps, "final_eps": final_eps}

dqn_optimizer_spec = dqn_OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),)

pg_optimizer_spec = pg_OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=learning_rate),)

# deep_Q_learning(env, dqn_architecture, dqn_optimizer_spec, encode_type,
#                 exploration_params, replay_buffer_size, start_learning,
#                 batch_size, gamma, target_update_freq, save_fig)

policy_gradient(env, pg_architecture, pg_optimizer_spec, learning_rate, encode_type,
                    num_of_episodes, gamma=0.99, save_fig=True)
