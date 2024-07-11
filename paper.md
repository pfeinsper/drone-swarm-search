---
title: 'DSSE: An environment for simulation of reinforcement learning-empowered drone swarm maritime search and rescue missions'
tags:
  - Python
  - PettingZoo
  - reinforcement learning
  - multi-agent
  - drone swarms
  - maritime search and rescue
  - shipwrecked people
authors:
  - name: Renato Laffranchi Falcão
    orcid: 0009-0001-5943-0481
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Jorás Custódio Campos de Oliveira
    orcid: 0009-0005-1883-8703
    equal-contrib: true
    affiliation: 1
  - name: Pedro Henrique Britto Aragão Andrade
    orcid: 0009-0000-0056-4322
    equal-contrib: true
    affiliation: 1
  - name: Ricardo Ribeiro Rodrigues
    orcid: 0009-0008-1237-3353
    equal-contrib: true
    affiliation: 1
  - name: Fabrício Jailson Barth
    orcid: 0000-0001-6263-121X
    equal-contrib: true
    affiliation: 1
  - name: José Fernando Basso Brancalion
    orcid: 0000-0002-4387-0204
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Insper, Brazil
   index: 1
 - name: Embraer, Brazil
   index: 2
date: 29 April 2024
bibliography: paper.bib

---

# Summary

The goal of this project is to advance research in maritime search and rescue missions using Reinforcement Learning techniques. The software provides researchers with two distinct environments: one simulates shipwrecked people drifting with maritime currents, creating a stochastic setting for training and evaluating autonomous agents; the other features a realistic particle simulation for mapping and optimizing search area coverage by autonomous agents.

Both environments adhere to open-source standards and offer extensive customization options, allowing users to tailor them to specific research needs. These tools enable Reinforcement Learning agents to learn efficient policies for locating shipwrecked individuals or maximizing search area coverage, thereby enhancing the effectiveness of maritime rescue operations.

# Statement of need

Maritime navigation plays a crucial role across various domains, including leisure activities and commercial fishing. However, maritime transportation is particularly significant as it accounts for 80% to 90% of global trade [@allianz]. While maritime navigation is essential for global trade, it also poses significant safety risks, as evidenced by the World Health Organization's report [@who] of approximately 236,000 annual drowning deaths worldwide. Therefore, maritime safety is essential, demanding significant enhancements in search and rescue (SAR) missions. It is imperative that SAR missions minimize the search area and maximize the chances of locating the search object.

To achieve this objective, traditional SAR operations have utilized path planning algorithms such as parallel sweep, expanding square, and sector searches [@iamsar]. However, these methods have not been optimal. Trummel & Weisinger [@trummel1986] demonstrated that finding an optimal search path, where the agent must search all sub-areas using the shortest possible path, is NP-complete. Recent research, however, proposes a different approach using Reinforcement Learning (RL) algorithms instead of pre-determined search patterns [@AI2021110098; @WU2024116403]. This is based on the belief that RL can develop new, more efficient search patterns tailored to specific applications. The hypothesis is that maximizing reward fosters generalization abilities, thereby creating powerful agents [@SILVER2021103535]. Such advancements could potentially save more lives.

The two primary metrics for evaluating an efficient search are coverage rate and time to detection. Coverage rate is the proportion of the search area covered by the search units over a specific period. Higher coverage rates typically indicate more effective search strategies. Time to detection is the time taken from the start of the search operation to the successful detection of the target. Minimizing this time is often a critical objective in SAR missions.

Expanding on the state-of-the-art research presented by @AI2021110098 and @WU2024116403, this project introduces a unique simulation environment that has not been made available by other researchers. Additionally, this new environment enables experiments on search areas that are significantly larger than those used in existing research.

# Functionality

In order to contribute to research on the effectiveness of integrating RL techniques into SAR path planning, the Drone Swarm Search Environment (`DSSE`), distributed as a Python package, was designed to provide a training environment using the PettingZoo [@terry2021pettingzoo] interface. Its purpose is to facilitate the training and evaluation of single or multi-agent RL algorithms. Additionally, it has been included as a third-party environment in the official PettingZoo documentation [@Terry_PettingZoo_Gym_for].

![Simulation environment showcasing the algorithm's execution.\label{fig:example}](docs/public/pics/dsse-example.png){ width=50% }

The environment depicted in \autoref{fig:example} comprises a grid, a probability matrix, drones, and an arbitrary number of persons-in-water (PIW). The movement of the PIW is influenced by, but not identical to, the dynamics of the probability matrix, which models the drift of sea currents impacting the PIW [@WU2023113444]. The probability matrix itself is defined using a two-dimensional Gaussian distribution, which expands over time, thus broadening the potential search area. This expansion simulates the diffusion of the PIW, approximating the zone where drones are most likely to detect them. Moreover, the environment employs a reward function that incentivizes the speed of the search, rewarding the agents for shorter successful search durations.

The package also includes a second environment option. Similar to the first, this alternative setup is designed for training agents, but with key differences in its objectives and mechanics. Unlike the first environment, which rewards agents for speed in their searches, this second option rewards agents that cover the largest area without repetition. It incorporates a trade-off by using a stationary probability matrix, but enhances the simulation with a more advanced Lagrangian particle model [@gmd-11-1405-2018] for pinpointing the PIW's position. Moreover, this environment omits the inclusion of shipwrecked individuals, focusing instead on promoting research into how agents can learn to efficiently expand their search coverage over broader areas.

Using this environment, any researcher or practitioner can write code and execute an agent's training, such as the source code presented below.

```python
from DSSE import DroneSwarmSearch

env = DroneSwarmSearch()

observations, info = env.reset()

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
```

The grid is divided into square cells, each representing a quadrant with sides measuring 130 meters in the real world. This correlation with real-world dimensions is crucial for developing agents capable of learning from realistic motion patterns. The drones, which are controlled by RL algorithms, serve as these agents. During the environment's instantiation, users define the drones' nominal speeds. These drones can move both orthogonally and diagonally across the grid, and they are equipped to search each cell for the presence of the PIW.

Several works have been developed over the past few years to define better algorithms for the search and rescue of shipwrecks [@AI2021110098; @WU2024116403]. However, no environment for agent training is made available publicly. For this reason, the development and provision of this environment as a Python library and open-source project are expected to have significant relevance to the machine learning community and ocean safety.

This new library makes it possible to implement and evaluate new RL algorithms, such as Deep Q-Networks (DQN) [@dqn2015] and Proximal Policy Optimization (PPO) [@ppo2017], with little effort. Additionally, several state-of-the-art RL algorithms have already been implemented and are available [@algorithmsDSSE2024]. An earlier iteration of this software was utilized in research that compared the Reinforce algorithm with the parallel sweep path planning algorithm [@dsse2023].

# References
