import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actor,self).__init__()
        self.fc1 = nn.Linear(state_dim, 2000)
        self.fc2 = nn.Linear(2000,action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 2000)
        self.fc2 = nn.Linear(2000,1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # print(x.shape)
        # print(len(y.shape))
        if len(y.shape) == 2:
            y = torch.unsqueeze(y, dim=1)
        x = torch.cat((x,y), 2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def put(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

import torch.optim as optim
import torch.nn.functional as F
class DDPG(object):
    def __init__(self, state_dim, action_dim, gamma = 0.99, tau = 0.001, lr_actor = 0.0001, lr_critic = 0.001, capacity = 500000, batch_size = 256):
        # 使用doublenet策略
        self.Actor = actor(state_dim, action_dim)
        self.Actor_Target = actor(state_dim, action_dim)
        self.Actor_Target.load_state_dict(self.Actor.state_dict())

        self.Critic = critic(state_dim, action_dim)
        self.Critic_Target = critic(state_dim, action_dim)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.optimizer_actor = optim.Adam(self.Actor.parameters(), lr_actor)
        self.optimizer_critic = optim.Adam(self.Critic.parameters(), lr_critic)
        self.memory = ReplayBuffer(capacity = capacity)
        self.batch_size = batch_size
        self.max_episode = 3
        self.max_step = 150

    def select_action(self, state, episode, action_space):
        if random.random() <= episode / self.max_episode + 0.2:
            action = self.Actor(state).cpu().detach().numpy()[0]
        else:
            action = action_space.sample()
        return action

    def update(self):

        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # 主网络网络更新
        ## 更新Critic：
        ### 计算损失
        Q_target = reward.unsqueeze(-1) + self.gamma * self.Critic_Target(next_state, self.Actor_Target(next_state))
        Q_current = self.Critic(state, action)
        Critic_Loss = F.mse_loss(Q_current, Q_target.detach())
        ### 更新参数
        self.optimizer_critic.zero_grad()
        Critic_Loss.backward()
        self.optimizer_critic.step()

        ## 更新Actor
        ### 计算损失
        Actor_Loss = - self.Critic(state, self.Actor(state)).mean()
        ### 更新参数
        self.optimizer_actor.zero_grad()
        Actor_Loss.backward()
        self.optimizer_actor.step()

        # 更新目标网络
        self.softupdate(self.Actor, self.Actor_Target)
        self.softupdate(self.Critic, self.Critic_Target)

    def softupdate(self, local_model, target_model):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)

# 算法测试，cache环境
import numpy as np
import torch
from AFRL_ENV_Google import Edge_Caching_Vehicular_Networks

# 创建环境
env = Edge_Caching_Vehicular_Networks()
env.algorithm = 'AFRL'
env.show = True
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 设置智能体
agent = DDPG(state_dim, action_dim)
agent.Actor = agent.Actor.to(device)
agent.Actor_Target = agent.Actor_Target.to(device)
agent.Critic = agent.Critic.to(device)
agent.Critic_Target = agent.Critic_Target.to(device)

# 训练智能体
for i_episode in range(agent.max_episode):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    for t in range(agent.max_step):
        action = agent.select_action(state, i_episode, env.action_space)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        agent.memory.put(state, action, reward, next_state, done)
        agent.update()
        if (t+1)%200 == 0:
            print(F'({(t+1)//200}). hit_ratio: {round(env.hit_ratio,3)}; service_delay: {round(sum(env.service_delay) / max(1, len(env.service_delay)),3)}')
        state = next_state
        if done:
            break

    print(f'Episode {i_episode + 1}: Finished after {t + 1} timesteps')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15,3))
    axs[0].plot(range(len(env.service_delay[::10])),env.service_delay[::10],color = 'brown')
    axs[0].set_title('service delay')
    axs[1].plot(range(len(env.hit_ratio_LST) - 5),env.hit_ratio_LST[5:],color='blue')
    axs[1].set_title('hit ratio')
    plt.show()

import pickle
with open("agent-1.pickle", 'wb') as f:
    pickle.dump(agent, f)