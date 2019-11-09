[//]: # (Image References)

[image1]: ddpg_agent_learning_curve.jpg "Learning Curve"
[image2]: ddpg_agent_learning_curve_BatchSize4096.jpg "Learning Curve for BatchSize 4096"
[image3]: ddpg_agent_learning_curve_SingleAgent.jpg "Learning Curve for training with Single Agent"

# Project 1: Navigation

# Report

## The Learning Algorithm
I implemented the Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)) algorithm to train the agent. DDPG is a model-free, off-policy actor-critic algorithm using deep function approximators  that can learn high-dimensional, continous action spaces.

The pseudo-code for the algotithm is as follows (taken from [DDPG](https://arxiv.org/abs/1509.02971)) paper:

    Randomly initialize critic network Q(s,a|θQ) and actor μ(s|θμ) with weights θQ and θμ
    Initialize target network Q′ and μ′ with weights θQ′ <-- θQ, θμ′ <-- θμ
    Initialize replay buffer R
    for episode = 1, M do
        Initialize a random process N for action exploration
        Receive initial observation state s1
        for t = 1, T do
            Select action at = μ(st|θμ) + Nt according to the current policy and exploration noise
            Execute action at and observe reward rt and observe new state st+1
            Store transition (st,at,rt,st+1) in R
            Sample a random minibatch of NN transitions(si,ai,ri,si+1) from R
            Set yi = ri + γ Q′(si+1, μ′(si+1|θμ′)|θQ′)
            Update critic by minimizing the loss: L = 1/N ∑i (yi− Q(si,ai|θQ))**2
            Update the actor policy using the sampled policy gradient: ∇θμ J ≈ 1/N ∑i ∇a Q(s,a|θQ)|s=si,a=μ(si) ∇θμ μ(s|θμ)|si
            Update the target networks:
                θQ′←τ θQ+ (1−τ) θQ′
                θμ′←τ θμ+ (1−τ) θμ′
        end for
    end for

My implementation uses to following hyperparameter values:

* Replay Memory capacity: 1,000,000 experiences
* Batch size: 1024 experiences
* gamma: 0.99
* tau: 0.001
* Learning Rate (actor): 0.0001
* Learning Rate (critic): 0.0003
* 10 learning steps for every 400 experiences collected (in the multi agent environment, 20 agents collect 400 experiences in 20 steps)

## The neural network models

The Actor and the Critic each have "local" and "target" neural networks.

The Actor networks are a typical neural network consisting each of three fully connected layers. There are 33 input nodes (number of states) and four output nodes (number of actions). The two hidden layers have 256 and 128 nodes, respectively, and a Leaky ReLU activation function. The output layer has a tanh function so that the output range is [-1, +1].

The Critic networks are a classic neural network consisting each of four fully connected layers. There are 37 input nodes (number of states + number of actions) and one output node. The three hidden layers have 256, 128, and 128 nodes, respectively, and a Leaky ReLU activation function.

The weights of the target Actor and Critic networks are updated using soft updates.


## The Learning Curve

I ran several experiments with the models, to see the impact of different hyperparameter values. Most experiments showed either slower learning than the final network presented above, or no learning, and were aborted before the target score was reached.

Below is the learning curve for the model presented above. The DDPG algorithm ran for 309 episodes, taking 3 hours and 40 minutes on my iMac (CPU only), until the average of 100 consecutive episode scores exceeded 30. This means that the environment was sovled in 209 episodes.

![Learning Curve][image1]

When the batch size was increased to 4096, the environment was solved in 116 episodes, much less then in the scenario presented above where the Batch Size was 1024. Yet it took 9 hours and 24 minutes to complete on my iMac. The learning curve for this experiment is shown below.

![Learning Curve][image2]

I also experimented with using the Single Agent version of the environment. Using the same hyperparameters as presented above, the environment was solved in 3073 episodes and it took 4 hours and 44 minutes on my MacBookPro, which is slightly slower than my iMac. The learning curve for this experiment is shown below.

![Learning Curve][image3]

I also ran experiments using GPU in the Udacity Workspace and on Amazon AWS. The speed-up gains where not significant or negative. This may be because the neural network model is not very large, or because my code is not tuned to run efficiently on GPU.

## Further Improvements

The following improvements to DQN could accelerate the learning process:
* Double DQN
* Prioritized Experience Replay
* Dueling DQN
* Learning from multi-step bootstrap targets
* Distributional DQN
* Noisy DQN
* Rainbow: all the above combined !
