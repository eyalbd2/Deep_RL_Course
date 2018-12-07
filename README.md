# Deep_RL_Course

In this project we solve two RL environments: "taxi-v2" and "acrobot-v0". We use Deep Reinforcement Learning methods to solve these tasks and we present our results in detail in the attached document "Deep_RL_HW".

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/Images/master/acrobot-taxi-image.png" width="400" title="Acrobot Taxi">
</p>


The project implementation contain:
1. DQN - solving Taxi env using both agent and target neural network, encoding the state to a 'one hot' input to the net. We explore also another input encoding, using taxi position, person position and final position for dropout.   
2. Actor Critic - we use actor critic algorithm to lear a policy for the taxi environment. The policy is a probability taking a specific action given a current state. We suggest a fully connected network with two output head, one to learn
directly the policy π(a|s) and the other to estimate the value function n Vˆ(s), as illustrated as illustrated below.
3. Prioritized Experinecr Replay - we use PER method in order to speed up agent training session on acrobot task.
4. DQN using CNN - to solve acrobot task we encode the current state to be a difference image between the current game board to the last game board, hence we use a convolutional neural network to learn Q value efficiently.
5. We analyze our result, explore architectural parameters such as using dropout, max pooling, hidden state dimension, input different encoding (both for taxi and for acrobot), normalization relevance, etc.


<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/Images/master/actor_critic_image.JPG" width="400" title="Actor Critic Net">
</p>

## Getting Started

- [ ] install and arrange an environment that meets the prerequisites (use below subsection)
- [ ] we recommand installing pycharm and using it to further use the code 
- [ ] clone this directory to your computer
- [ ] you can use test codes to demonstrate the agent behavior per each task
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
This project solves two tasks, hence for running a test the user should enter on of the sub-directoris ('taxi'\'acrobot') and run tests from there. 

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

## Results for tests
We have constructed some exrepiments for hyper parameters calibration and for study.
All results are shown in detail in the report we wrote, which is also in the main directory "Deep_RL_Course".
We also present below some of our main experiments results.




## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
