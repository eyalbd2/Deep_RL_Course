# Advanced Topics in Reinforcement Learning (048716) 

In this project we solve two RL environments: "taxi-v2" and "acrobot-v1". For solving those tasks, we use Deep Reinforcement Learning methods with our results presented in detail in the attached document "Deep_RL_HW".

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/acrobot-taxi-image.png" width="400" title="Acrobot Taxi">
</p>


The project implementation contains:
1.	DQN - solving Taxi env using both agent and target neural network, encoding the state to a 'one hot' input to the net. Additionally we explore another input encoding, using taxi position, person position and final position for dropout.
2.	Actor Critic - we use actor critic algorithm to learn a policy for the taxi environment. The policy is a probability taking a specific action in a current state. We suggest a fully connected network with two output heads. The first, to learn directly the policy π(a|s) and the other to estimate the value function n V(s), as illustrated below.
3.	Prioritized Experience Replay - we use PER method in order to speed up agent training session on acrobot task.
4.	DQN using CNN – In order to solve the acrobot task we encode the current state image as the difference between the current game board to the last game board, hence we use a convolutional neural network to learn Q value efficiently.
5.	We analyze our results, explore architectural parameters such as using dropout, max pooling, hidden state dimension, input different encoding (both for taxi and for acrobot), normalization relevance, etc.



<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/actor_critic_image.JPG" width="400" title="Actor Critic Net">
</p>
Here we present the actor critic net with two outputs.

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/CNN_acrobot.JPG" width="400" title="Acrobot CNN">
</p>
This is an illustration of the CNN we used for DQN with images representing states as input to the net.

## Getting Started

- [ ] install and arrange an environment that meets the prerequisites (use below subsection)
- [ ] we recommend installing pycharm and using it to further use the code
- [ ] clone this directory to your computer
- [ ] you can use test codes to demonstrate the agent behavior per task
- [ ] use training codes to train the agent by yourself changing hyper parameters as you please

### Prerequisites

Before starting, make sure you already have the followings, if you don't - you should install before trying to use this code:
- [ ] the code was tested only on python 3.6 
- [ ] install gym
- [ ] conda/anaconda packages
- [ ] pytorch and torchvision - for implementation of FNN/CNN
- [ ] matplotlib - for plots and for input encoding in acrobot task
- [ ] numpy - for basic math and vectors operations


## Running the tests
This project solves two tasks, hence for running a test the user should enter on of the sub-directories ('taxi''acrobot') and run tests from there.

### Taxi
First enter "Taxi" directory.

#### DQN
To train the DQN agent use the command:
```
python DQN_train.py
```
And for teting the agent behaivior run the command:
```
python DQN_test.py
```

#### Actor Critic
To train the Actor Critic agent use the command:
```
python AC_train.py
```
And for teting the agent behaivior run the command:
```
python AC_test.py
```

### Acrobot
First enter "Acrobot" directory.

To train the DQN using PER agent use the command:
```
python DQN_train.py
```
And for teting the agent behaivior run the command:
```
python DQN_test.py
```

The presented video below is an actual result of the code above 
![alt text](./Images/acrobot_video.gif?raw=true)

## Results for tests
We have constructed some experiments for hyper parameters calibration and for study. All results are shown in detail in the report we wrote, which is also in the main directory "Deep_RL_Course". We also present below some of our main results.

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/DQN_Taxi.JPG" width="400" title="Taxi DQN Converge">
</p>
Presented above is the Average accumulated episodic reward against number of episodes for different hidden layer dimensions for the Taxi agent using DQN.

Next we will show the learning graph for the actor critic agent solvig taxi environment.
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/ActorCriticConvergence.JPG" width="400" title="Actor Critic Converge">
</p>
The accumulated episodic reward and its average over the most recent 10 episodes for the actor-critic model for both limiting and non- limited episode’s length to 200 transitions in (a) and (b), respectively. Notice the scale difference.
For the acrobot environment we present similar graphs as we demonstrated for actor critic. Here we used a PER, which speeded up the learning convergence.


For the acrobot environment we present similar graphs as we demonstrated for actor critic. Here we used a PER, which speeded up the learning convergence.
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/Images/acrobot_result_convergence.JPG" width="400" title="Acrobot DQN">
</p>
The accumulated episodic reward and its average over the most recent 10 episodes for the DQN model for the acrobot task presenting the episodic accumulated reward and its average over last 10 episodes for each step in (a) and the average episodic accumulated reward together with its std over 10 episodes in (b).




## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors

* **Ron Dorfman** 
* **Eyal Ben David** 


## License

This project is licensed under the MIT License 

## Acknowledgments

* DQN Basic Implementation - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
* Sum Tree implementation - https://github.com/jaromiru/AI-blog
* Actor Critic implementation - 
* Good easy explanation of PER - https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
