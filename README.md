# Deep_RL_Course

In this project we solve two RL environments: "taxi-v2" and "acrobot-v0". We use Deep Reinforcement Learning methods to solve these tasks and we present our results in detail in the attached document "Deep_RL_HW".
![](https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/acrobot-taxi-image.png)

The project implementation contain:
1. DQN - solving Taxi env using both agent and target neural network, encoding the state to a 'one hot' input to the net. We explore also another input encoding, using taxi position, person position and final position for dropout.   
2. Actor Critic - we use actor critic algorithm to lear a policy for the taxi environment. The policy is a probability taking a specific action given a current state. We suggest a fully connected network with two output head, one to learn
directly the policy π(a|s) and the other to estimate the value function n Vˆ(s), as illustrated as illustrated below.

![alt text](https://raw.githubusercontent.com/eyalbd2/Deep_RL_Course/master/actor_critic_image.JPG)




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
