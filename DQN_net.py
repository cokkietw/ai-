# 使用DQN网络，一帧一帧传入，基于价值选取动作,加入经验回放，利用经验池更新网络参数

import gym
import time
import cv2 as cv
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
env = gym.make('Pong-v4')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN网络
class DQN_Q_Net(nn.Module):
    def __init__(self, n_actions=6):
        super(DQN_Q_Net,self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # 全连接神经网络
        self.linear1 = nn.Linear(1024,256)
        self.linear2=nn.Linear(256,64)
        self.linear3 = nn.Linear(64,n_actions)
        # 激活函数
        self.ReLU = nn.ReLU() 
        self.Sigmoid=nn.Sigmoid()
        self.Softmax=nn.Softmax(-1)
    def forward(self, x):# 前向传播
        x = self.conv1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.pool(x)

        # 对每一帧图像压平
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.ReLU(x)
        
        x = self.linear2(x)
        x = self.ReLU(x)
 
        output=self.linear3(x)
        return output

class Breakout_agent:
    def __init__(self,env,action_n=6,gamma=0.99, batch_size=64,epsilon=0.01,
                 learning_rate=0.0001,train_episode=5000,test_episode=5000,
                 replay_buffer_size=10000):
        # 智能体参数
        self.learning_rate=learning_rate
        self.train_episode=train_episode
        self.test_episode=test_episode
        self.action_n=action_n
        self.epsilon=epsilon
        self.batch_size = batch_size
        self.gamma=gamma
        self.count=0
        self.replay_buffer_size=replay_buffer_size

        # 智能体网络
        self.net=DQN_Q_Net(action_n).to(device)
        self.target_net=DQN_Q_Net(action_n).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # 经验池
        self.replay_buffer=deque([],self.replay_buffer_size)

    # 基于价值选取动作
    def decide(self,state):
        state=state.to(device)
        
        if random.random() >self.epsilon:
            with torch.no_grad():
                action=self.net.forward(state)
            return action.argmax()
        else:
            return torch.tensor(np.random.randint(self.action_n)).to(device)
        
    def learn(self):
        if len(self.replay_buffer)<self.batch_size:
            return 
        experiences = random.sample(self.replay_buffer,self.batch_size)
        states,actions,rewards,next_states,dones=zip(*experiences)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions).unsqueeze(1)
        dones = torch.stack(dones).unsqueeze(1)
        rewards = torch.stack(rewards).unsqueeze(1)
        # 计算Q值
        Q_value = self.net(states).gather(1,actions)
        with torch.no_grad():
            next_Q_value = self.target_net(next_states).max(1)[0].view(-1,1)
        new_Q_value =rewards + self.gamma * next_Q_value
        loss = torch.mean(nn.functional.mse_loss(Q_value,new_Q_value))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count % 3000 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.count=0
        self.count+=1
        return loss
        
    # 保存和读取历史网络参数
    def save(self, filename='dqn_model.pth'):
        torch.save(self.net.state_dict(), filename)
        print(f"Model saved to {filename}")
    def load(self, filename='dqn_model.pth'):
        self.net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))
        self.net.eval()
        self.target_net.eval()
        print(f"Model loaded from {filename}")

# 状态处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image 

# 训练函数
def train(agent,env):
    # env.render()
    episode_reward=0
    state=env.reset()
    state=preprocess(state)
    state=np.array(state)
    state=torch.from_numpy(state).float()
    # state=torch.flatten(state)
    state=state.to(device).unsqueeze(0)
    while True:
        env.render()
        action=agent.decide(state.unsqueeze(0))
        
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
        next_state=np.array(next_state)
        next_state=torch.from_numpy(next_state).float()
        # next_state=torch.flatten(next_state)
        next_state=next_state.to(device).unsqueeze(0)
        reward = torch.tensor(reward).to(device)
        done = torch.tensor(done).to(device)
        agent.replay_buffer.append([state,action,reward,next_state,done])
        agent.learn()
        episode_reward+=reward
        # agent.save()
        if done:
            # agent.save()
            break
        state = next_state
    episode_reward = int(episode_reward)
    return episode_reward

# 测试函数
def test(agent,env):
    # env.render()
    episode_reward=0
    state=env.reset()
    state=preprocess(state)
    state=np.array(state)
    state=torch.from_numpy(state).float()
    # state=torch.flatten(state)
    state=state.to(device).unsqueeze(0)
    while True:
        env.render()
        action=agent.decide(state.unsqueeze(0))
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
        next_state=np.array(next_state)
        next_state=torch.from_numpy(next_state).float()
        # next_state=torch.flatten(next_state)
        next_state=next_state.to(device).unsqueeze(0)
        episode_reward+=reward
        # agent.save()
        if done:
            # agent.save()
            break
        state = next_state
    return episode_reward

# 绘制折线图
def draw(rewards):
    plt.plot([i for i in range(len(rewards))], rewards)
    plt.xlabel('episode')
    plt.title('result')
    plt.ylabel('reward')
    plt.show()


agent=Breakout_agent(env)
agent.load()

episode_rewards=[]
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    if (episode+1) % 50 ==0:
        agent.save()
draw(episode_rewards)

agent.save()

episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)
