# 使用策略梯度网络，一局一局游戏传入，基于概率选择动作，加入奖励衰减机制处理奖励，最大化概率*奖励以更新策略网络
# Gym-Pong游戏
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
        ##### 计算概率
        output = self.Softmax(output)
        return output

class Breakout_agent:
    def __init__(self,env,action_n=6,gamma=0.99, batch_size=64,
                 learning_rate=0.0001,train_episode=5000,test_episode=5000,
                 replay_buffer_size=10000):
        # 智能体参数
        self.learning_rate=learning_rate
        self.train_episode=train_episode
        self.test_episode=test_episode
        self.action_n=action_n
        self.batch_size = batch_size
        self.gamma=gamma
        self.count=0
        self.replay_buffer_size=replay_buffer_size

        # 智能体网络
        self.net=DQN_Q_Net(action_n).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
    # 基于概率选取动作
    def decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.net.forward(state)
        probs=probs.squeeze(0)
        probs = probs.cpu().numpy()
        return np.random.choice(len(probs),1,p=probs)
    # 测试时，选取最大概率的动作
    def test_decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.net.forward(state)
        probs=probs.squeeze(0)
        probs = probs.cpu().numpy()
        return probs.argmax().numpy() 
    # 奖励衰减函数
    def calc_reward_to_go(self,reward_list, gamma=0.99):
        reward_arr = np.array(reward_list)
        for i in range(len(reward_arr) - 2, -1, -1):
            # 递推衰减
            reward_arr[i] += gamma * reward_arr[i + 1]
        # 标准化奖励
        reward_arr -= np.mean(reward_arr)
        reward_arr /= np.std(reward_arr)
        reward_arr = torch.from_numpy(reward_arr).to(device)
        return reward_arr
    # 学习函数
    def learn(self,states,actions,rewards):
        probs = self.net(states)
        rewards = self.calc_reward_to_go(rewards)
        actions = torch.stack(actions).T
        log_probs = torch.distributions.Categorical(probs).log_prob(actions)
        self.optimizer.zero_grad()
        loss = -torch.mean(log_probs * rewards)
        loss.backward()                                                                                                                                                              
        self.optimizer.step()
        return loss
    # 保存和读取历史网络参数
    def save(self, filename='p_net_model.pth'):
        torch.save(self.net.state_dict(), filename)
        print(f"Model saved to {filename}")
    def load(self, filename='p_net_model.pth'):
        self.net.load_state_dict(torch.load(filename))
        self.net.eval()
        print(f"Model loaded from {filename}")

# 状态处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    image=np.array(image)
    image=torch.from_numpy(image).float()
    image=image.to(device)
    image = image.unsqueeze(0)
    return image 


# 训练函数
def train(agent,env):
    # env.render()
    episode_reward=0
    state=env.reset()
    state=preprocess(state)
    frames, actions, rewards = [], [], []
    while True:
        frames.append(state)
        env.render()
        action=agent.decide(state.unsqueeze(0))
        action = torch.tensor(action).to(device)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
        rewards.append(reward)
        episode_reward+=reward
        # agent.save()
        if done:
            # agent.save()
            break
        state = next_state
    episode_reward = int(episode_reward)
    loss=agent.learn(torch.stack(frames),actions,rewards)
    print(loss)
    return episode_reward

# 测试函数
def test(agent,env):
    # env.render()
    episode_reward=0
    state=env.reset()
    state=preprocess(state)
    while True:
        env.render()
        action=agent.test_decide(state.unsqueeze(0))
        action = torch.tensor(action).to(device)
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
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
# agent.load()

episode_rewards=[]
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    if episode % 50 ==0:
        agent.save()
# draw(episode_rewards)

agent.save()

episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)
