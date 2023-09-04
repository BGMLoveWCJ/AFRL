# AFRL
This repository is the code for the paper entitled "Deep reinforcement Learning with Asynchronous Federated Learning for Vehicular Edge Data Caching" [Unsubmitted]
(Author: Guanming Bao)

Abstract: Edge computing is a distributed computing paradigm that relocates cloud-like computation and data storage services closer to users. The edge caching is a promising tech- nology to support rapidly growing diversified vehicular services and applications. The content and service caching on edge nodes enables the quick content delivery and reduces the service la- tency. To achieve high caching efficiency, learning-based caching algorithms have been presented in the literature. However, to maintain learning-based caching algorithms practicable, it is critical to keep communication and computing costs minimal, while also ensuring the security of vehicle clients’ private data. To address these issues, this work presents AFRL, an Asyn- chronous Federated learning with deep Reinforcement Learning edge caching framework for vehicular networks. Specifically, we introduce federated learning to enable a group of vehicle clients to collaboratively train a shared DRL agent, reducing communication cost while protecting user privacy. The DRL agent is used to learn the optimal caching policy in the dynamic environment. To accelerate convergence to optimal solution, we design an efficient asynchronous federated learning framework that enables vehicle clients to perform immediate global updates for the next round of local training without having to wait for all local models to be uploaded. Simulation results demonstrate that the proposed AFRL outperforms traditional caching methods in terms of cache hit rate. Further, when compared to the synchronous baseline technique, AFRL converges to optimal solution faster and achieves higher cache hit rate in environments with varying traffic densities.

Architecture:\
![Architecture](https://github.com/BGMLoveWCJ/AFRL/blob/main/Thoughts/1.png)

Train step:\
![Train](https://github.com/BGMLoveWCJ/AFRL/blob/main/Thoughts/2.png)

Some simulation samples:\
![1RSU](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/1-_online-video-cutter.com_.gif)
![3RSU](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/3-_online-video-cutter.com_.gif)

Simulation description: The vehicle clients request media content from RSU(Blue lines in the samples), and the RSU check if the requested content is cached locally, when hit on the current RSU then the requested content will be provided by the current RSU(Red lines in the samples). The RSU will ask for help from the neighbour RSUs, if they cache the requested content, they will send it the the current RSU and provide the cache service. When the request misses both on the current RSU and its neighbours, the request will be relayed to the CDC, and the cache services will be provided by it(Pink lines in the samples).

Some tmp results:\
Random->10.2%(hit ratio)\
FRL->10.7%(hit ratio)\
![rst1](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/rst1.png)
![rst2](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/rst2.png)

To do:\
[1] Try some better asynchronous aggregation function.\
[2] Consider stroger DRL algorithms like TD3 and Rainbow.\
[3] Consider enabling cooperation between RSUs(MADDPG/Global Critic and some worker actors).
