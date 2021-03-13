# Report
---

## Video 

[![](http://img.youtube.com/vi/C5LOgWMtFrY/0.jpg)](http://www.youtube.com/watch?v=C5LOgWMtFrY "")

## Learning algorithm

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

The task is episodic, and in order to solve the environment, agent must get an average score of +0.5 over 100 consecutive episodes. 
Training algorithm is `In [5]: ddpg` inside [Tennis.ipynb](https://github.com/AInitikesh/DRLND-p3_collab-compet/blob/main/Tennis.ipynb). This function iterates over `n_episodes=3000` to train the ddpg agent model. After 3000 episodes model was not learning much and average score was decreasing so it doesn't makes sense to train the Agent after 3000 steps. I have created two separate instances of ddpg agents each agent has its own Replay buffer, Actor and critic networks.

### DDPG Agent Hyper Parameters

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZE (int): minibatch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR_ACTOR (float): learning rate of the actor 
- LR_CRITIC (float): learning rate of the critic
- WEIGHT_DECAY (float): L2 weight decay

Where 
`BUFFER_SIZE = int(1e6)`, `BATCH_SIZE = 128`, `GAMMA = 0.99`, `TAU = 1e-3`, `LR_ACTOR = 8e-5`, `LR_CRITIC = 8e-5` and `WEIGHT_DECAY = 0`   

### Neural Network

DDPG is an actor-critic method which uses 2 neural networks. One is Actor network that learns to predicts best action for given state and Critic network that learns to estimate the Q value for a given state action pair. Critic network is used to estimate the loss for Actor network.

1) [Actor model](https://github.com/AInitikesh/DRLND-p3_collab-compet/blob/main/model.py#L12) - Consist of an input layer of state size(24), two fully connected hidden layers of size 200 and 150 having relu activation and output fully connected layer size of action_size(2) and tanh activation function.

1) [Critic model](https://github.com/AInitikesh/DRLND-p3_collab-compet/blob/main/model.py#L44) - Consist of two input layers. First  input of state size(24) followed by fully connected hidden layer of size 200 and relu activation. We then concat the output of first hidden layer with second input ie action size(2) followed by one more hidden layer of size 150 and relu activation. Final output layer predicts a single Q value.

![DDPG algorithm](https://github.com/AInitikesh/DRLND-p3_collab-compet/blob/main/ddpg-algo.png)

Referenced from original paper [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1509.02971v6.pdf)

## Plot of Rewards

### Reward Plot QNetwork

![Reward Plot DDPG Network](https://github.com/AInitikesh/DRLND-p3_collab-compet/blob/main/score-card.png)

```
Episode 100	Average score: 0.015
Episode 200	Average score: 0.036
Episode 300	Average score: 0.035
Episode 400	Average score: 0.028
Episode 500	Average score: 0.029
Episode 600	Average score: 0.038
Episode 700	Average score: 0.040
Episode 800	Average score: 0.038
Episode 900	Average score: 0.038
Episode 1000	Average score: 0.036
Episode 1100	Average score: 0.042
Episode 1200	Average score: 0.048
Episode 1300	Average score: 0.054
Episode 1400	Average score: 0.055
Episode 1500	Average score: 0.081
Episode 1600	Average score: 0.084
Episode 1700	Average score: 0.089
Episode 1800	Average score: 0.097
Episode 1900	Average score: 0.094
Episode 2000	Average score: 0.101
Episode 2100	Average score: 0.106
Episode 2200	Average score: 0.141
Episode 2300	Average score: 0.139
Episode 2400	Average score: 0.145
Episode 2500	Average score: 0.180
Episode 2600	Average score: 0.201
Episode 2700	Average score: 0.354
Solved in episode: 2778 	Average score: 0.509
```

## Ideas for Future Work

Implement other methods like Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG) having common replay buffer for both the agents. 

Also adding batch normalisation layers and tuning hyper parameters of neural network architecture could help.