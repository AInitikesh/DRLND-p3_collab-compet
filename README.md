[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image3]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Implemented method

We have used ddpg agent to solve this environment and solved the environment in 100 episodes with Average score 0.509. More details the learning algorithm, along with the chosen hyper parameters is mentioned in Report.md. It also describes the model architectures for neural networks.

### Getting Started

1. Clone the repository using following command.

    ```
    git clone https://github.com/AInitikesh/DRLND-p3_collab-compet.git
    ```

2. To set up your Python environment correctly follow [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the DRLND GitHub repository, in the `DRLND-p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instruction to run the code.

1. Easiest way to replicate the results is to launch the jupyter notebook environment using following command in the root directory `DRLND-p3_collab-compet/`.
    ```
    jupyter notebook
    ```

2. Open `Tennis.ipynb` notebook from the jupyter environment in browser.

3. Before running code in a notebook, change the `kernel` to match the `drlnd` environment by using the drop-down Kernel menu.
    
    ![Kernel][image3]

4. Follow the instructions in `Tennis.ipynb` to get started with training your own agent. If you wish to skip training and just want to see the results from pre-trained models then ignore section `4.Training the ddpg agent` from `Tennis.ipynb`.

5. If you want to explore the code further.
    1. `ddpg_agent.py` contains code for ddpg agent.
    2. `model.py` contains neural network code required by the agent. 
    3. `checkpoint_actor0.pth`,  `checkpoint_actor1.pth`, `checkpoint_critic0.pth` and `checkpoint_critic1.pth` are the pre-trained weights files for Actor and Critic networks.