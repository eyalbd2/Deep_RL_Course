import torch
import gym
from utils import get_screen

# Import environment
env = gym.make('Acrobot-v1')
# Define model and load pre-trained weights
trained_Q_network = torch.load('trained_DQN_model', map_location=lambda storage, loc: storage)

# Reset env
_ = env.reset()
current_screen = get_screen(env)
state = current_screen
# Present trained behaviour over episodes
num_test_episodes = 30

episodes_passed = 0
acc_episodic_reward = 0.0

while episodes_passed < num_test_episodes:
    # Choose action greedily
    action = trained_Q_network.select_action(state)
    # Act on env
    _, reward, done, _ = env.step(action)
    last_screen = current_screen
    current_screen = get_screen(env)
    next_state = current_screen - last_screen
    # Add to accumulative reward
    acc_episodic_reward += reward
    # When episode is done - reset and print
    if done:
        # Print acc. reward
        print('Episode {}\tAccumulated Reward: {:.2f}\t'.format(
            episodes_passed+1, acc_episodic_reward))
        # Update statistics
        episodes_passed += 1
        acc_episodic_reward = 0.0

        _ = env.reset()
        current_screen = get_screen(env)
        next_state = current_screen

    state = next_state