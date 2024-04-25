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
    orcid: 0000-0000-0000-0000
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
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Insper, Brazil
   index: 1
 - name: Embraer, Brazil
   index: 2
date: 19 April 2024
bibliography: paper.bib

---

# Summary

The goal of this project is to contribute to the research of solutions that employ reinforcement learning techniques to maritime search and rescue missions of shipwrecked people. The software equip's researchers with a simulation of shipwrecked people casted away according to maritime currents to producing a stochastic environment to be used to train and evaluate autonomous agents.

# Statement of need

Maritime navigation plays a crucial role across various domains, including leisure activities and commercial fishing. However, maritime transportation is particularly significant as it accounts for 80% to 90% of global trade [@allianz]. Therefore, maritime safety is essential, demanding significant enhancements in search and rescue (SAR) missions. It is imperative that SAR missions minimize the search area and maximize the chances of locating the search object.

To achieve this objective, traditional SAR operations used path planning algorithms such as parallel sweep, expanding square, and sector searches [@iamsar]. But recent researches propose a different approach to this problem using reinforcement learning techniques over pre-determined search patterns [@AI2021110098; @WU2024116403].

In order to contribute to researches on the effectiveness of integrating reinforcement learning techniques into SAR path planning, the `DSSE`, distributed as a Python package, was designed to provide a simulation environment using the PettingZoo interface with the purpose of training and evaluating single or multi-agent reinforcement learning algorithms.

![Simulation environment showcasing the algorithm's execution.\label{fig:example}](docs/pics/dsse-example.png){ width=50% }

The environment depicted in \autoref{fig:example} comprises a grid, a probability matrix, drones, and an arbitrary number of persons-in-water (PIW). The movement of the PIW is influenced by, but not identical to, the dynamics of the probability matrix, which models the drift of sea currents impacting the PIW [@WU2023113444]. Additionally, the probability matrix is defined using a two-dimensional Gaussian distribution. Over time, this distribution expands, effectively broadening the area where the PIW might be located. This simulated diffusion approximates the zone where drones are most likely to detect the PIW.

The grid is divided into square cells, each representing a quadrant with sides measuring 130 meters in the real world. This correlation with real-world dimensions is crucial for developing agents capable of learning from realistic motion patterns. The drones, which are controlled by reinforcement learning algorithms, serve as these agents. During the environment's instantiation, users define the drones' nominal speeds. These drones can move both orthogonally and diagonally across the grid, and they are equipped to search each cell for the presence of the PIW.

`DSSE` was originally proposed as part of a Capstone project at Insper in a partnership with Embraer, in order to research the viabillity of exploring and rescuing shipwrecked people with reinforcement learning-empowered drone swarms [@dsse2023]. Since there is no available environment for such purpose and considering that the multi-agent reinforcement learning field of study is recent, it was decided to continue with the development of the project, aiming to enhance even further the realism of the simulation and to promote the research.

Currently, the `DSSE` is being enhanced and used by students as part of a Capstone project at Insper in a partnership with Embraer to implement and evaluate new reinforcement learning algorithms such as Deep Q-Networks (DQN) [@dqn2015] and Proximal Policy Optimization (PPO) [@ppo2017]. Also, it is receiving further study on how to establish information exchanging mechanisms to increase the collaboration between the agents.

# Acknowledgements

We acknowledge intellectual contributions and mentoring from Prof. Dr. Fabrício J. Barth and from Dr. José Fernando B. Brancalion, who have been extremely enthusiastic and engaged with the advance of the project.

# References
