# AFRL
This repository is the code for the paper entitled "Deep reinforcement Learning with Asynchronous Federated Learning for Vehicular Edge Data Caching" [Unsubmitted]
(Author: Guanming Bao)

Abstract: Edge computing is a distributed computing paradigm that relocates cloud-like computation and data storage services closer to users. The edge caching is a promising tech- nology to support rapidly growing diversified vehicular services and applications. The content and service caching on edge nodes enables the quick content delivery and reduces the service la- tency. To achieve high caching efficiency, learning-based caching algorithms have been presented in the literature. However, to maintain learning-based caching algorithms practicable, it is critical to keep communication and computing costs minimal, while also ensuring the security of vehicle clients’ private data. To address these issues, this work presents AFRL, an Asyn- chronous Federated learning with deep Reinforcement Learning edge caching framework for vehicular networks. Specifically, we introduce federated learning to enable a group of vehicle clients to collaboratively train a shared DRL agent, reducing communication cost while protecting user privacy. The DRL agent is used to learn the optimal caching policy in the dynamic environment. To accelerate convergence to optimal solution, we design an efficient asynchronous federated learning framework that enables vehicle clients to perform immediate global updates for the next round of local training without having to wait for all local models to be uploaded. Simulation results demonstrate that the proposed AFRL outperforms traditional caching methods in terms of cache hit rate. Further, when compared to the synchronous baseline technique, AFRL converges to optimal solution faster and achieves higher cache hit rate in environments with varying traffic densities.

Architecture:\
![Architecture](https://github.com/BGMLoveWCJ/AFRL/blob/main/Thoughts/distributed%20architecture.png)

Train step:\
![Train]()

Samples:\
![1RSU](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/1-_online-video-cutter.com_.gif)
![3RSU](https://github.com/BGMLoveWCJ/AFRL/blob/main/demo/3-_online-video-cutter.com_.gif)
