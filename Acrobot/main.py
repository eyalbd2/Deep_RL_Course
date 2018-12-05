
import gym
import torch.optim as optim
from DQN import OptimizerSpec, deep_Q_learning


env = gym.make('Acrobot-v1')

input_width = 40
input_height = 40
input_depth = 3
hidden_units_0 = 32
hidden_units_1 = 32
hidden_units_2 = 64
kernel = 3
stride = 1
padding = 1
pool = 2
replay_buffer_size = 1000000
start_learning = 50000
# replay_buffer_size = 20
# start_learning = 20

learning_rate = 0.00025
alpha = 0.95
eps = 0.01
batch_size = 16
gamma = 0.99
target_update_freq = 500
schedule_timesteps = 1000000
final_eps = 0.1
num_of_episodes = 100000
save_fig = True

architecture = {"input_width": input_width, "input_height": input_height, "input_depth": input_depth,
                    "action_space": env.action_space.n, "hidden_units_0": hidden_units_0,
                    "hidden_units_1": hidden_units_1, "hidden_units_2": hidden_units_2, "kernel": kernel,
                    "stride": stride, "padding": padding, "pool": pool}

exploration_params = {"timesteps": schedule_timesteps, "final_eps": final_eps}

optimizer_spec = OptimizerSpec(constructor=optim.RMSprop,
                               kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),)



deep_Q_learning(env, architecture, optimizer_spec, exploration_params, replay_buffer_size, start_learning,
                batch_size, gamma, target_update_freq, save_fig)

