# Drone Swarm Search: Algorithms

## About

Welcome to our [Algorithms repository](https://github.com/pfeinsper/drone-swarm-search-algorithms)! These algorithms are specifically tailored for the environments available in `DSSE`, aimed at optimizing drone swarm coordination and search efficiency.

Explore a diverse range of implementations that leverage the latest advancements in machine learning to solve complex coordination tasks in dynamic and unpredictable environments. Our repository offers state-of-the-art solutions designed to enhance the performance and adaptability of autonomous drone swarms, making them more efficient and effective in various search and rescue missions.

<p align="center">
    <img src="/gifs/PPO_coverage.gif" width="400" height="400" align="center">
    <br>
    <em>Fig 1: Trained agents in action with PPO algorithm.</em>
</p>

## Algorithms used
All the algorithms developed here were done using the [RLlib library](https://docs.ray.io/en/latest/rllib/index.html)
Both implementations are done trough a Multi-Input Convolutional Neural Network, that receives the probability matrix as one of the inputs, and the agents positions as the other, in other words, the complete observations of the environment.

### Proximal Policy Optmization (PPO)
There are two implementations of the Proximal Policy Optimization (PPO) algorithm, a centralized and descentralized versions. The difference being that the centralized approach creates a single neural network for all the agents, while the descentralized version creates a neural network per-agent.


### Deep Q-Network (DQN)
There is a implementation for the Deep Q-Networks algorithm as well, using the same network architecture as the PPO version, changing only parts related to the output, that is specific to each of the algorithms.


## How to run

A variety of algorithm implementations are available in the algorithms directory. To run any script within this directory, use the following command:

```bash
python SCRIPT.py
```
exchanging `SCRIPT` with the desired script name.

Please note that configurations and parameters can be adjusted for both the environment and RLlib. Make sure to review and update these settings as needed to suit your specific requirements.

## Stay Updated

We appreciate your patience and interest in our work. If you have any questions or need immediate assistance regarding our `algorithms`, please do not hesitate to contact us via our [GitHub Issues page](https://github.com/pfeinsper/drone-swarm-search-algorithms/issues).

## License
This documentation is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for more details.
