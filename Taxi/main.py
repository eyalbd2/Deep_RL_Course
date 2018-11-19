
# Imports
import gym
import torch.optim as optim
from deep_q_learner import OptimizerSpec, deep_Q_learning


env = gym.make('Taxi-v2').unwrapped

hidden_units = 100
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
save_fig = True

architecture = {"num_states": env.observation_space.n, "hidden_units": hidden_units, "num_actions": env.action_space.n}
exploration_params = {"timesteps": schedule_timesteps, "final_eps": final_eps}

optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),)

deep_Q_learning(env, architecture, optimizer_spec,
                exploration_params, replay_buffer_size, start_learning,
                batch_size, gamma, target_update_freq, save_fig)
