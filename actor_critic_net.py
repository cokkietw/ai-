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
# DQN网络
class DQN_Net(nn.Module):
    def __init__(self):
        super(DQN_Net,self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # 全连接神经网络
        self.linear1 = nn.Linear(1024,256)
        self.linear2=nn.Linear(256,64)
        self.linear3 = nn.Linear(64,1)
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
    # def initialize_weights(self):
    #     for m in self.modules():
    #         nn.init.normal_(m.weight.data, 0, 0.1)
    #         nn.init.constant_(m.bias.data, 0.01)
# 策略网络
class Q_Net(nn.Module):
    def __init__(self, n_actions=6):
        super(Q_Net,self).__init__()
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
    # def initialize_weights(self):
    #     for m in self.modules():
    #         nn.init.normal_(m.weight.data, 0, 0.1)
    #         nn.init.constant_(m.bias.data, 0.01)

# 智能体
class Breakout_agent:
    def __init__(self,env,action_n=6,gamma=0.99, batch_size=64,
                 learning_rate=0.001,train_episode=5000,test_episode=5000,
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
        self.dqn = DQN_Net().to(device)
        self.act=Q_Net(action_n).to(device)
        self.target_dqn=DQN_Net().to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=self.learning_rate)

        # 经验池
        self.replay_buffer=deque([],self.replay_buffer_size)
    # 基于概率选取动作
    def decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.act.forward(state)
        probs=probs.squeeze(0)
        probs = probs.cpu().numpy()
        return np.random.choice(len(probs),1,p=probs)
    # 测试时，选取最大概率的动作
    def test_decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.act.forward(state)
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
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从缓冲区中抽样经验
        batch = random.sample(self.replay_buffer,self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        # DQN预测状态价值，用此更新评论家网络
        v = self.dqn.forward(states)
        with torch.no_grad():
            next_v = self.target_dqn.forward(next_states)
        reward = rewards.unsqueeze(1)
        dqn_loss = nn.functional.mse_loss(v,reward + self.gamma * next_v)
        self.dqn_optimizer.zero_grad()
        dqn_loss.backward()
        self.dqn_optimizer.step()
        
        # 更新演员网络
        with torch.no_grad():
            td_error = reward + self.gamma * next_v - v
        prob = self.act(states)
        log_probs = torch.distributions.Categorical(prob).log_prob(actions)
        self.act_optimizer.zero_grad()
        act_loss = -torch.mean(log_probs * td_error)
        act_loss.backward()
        self.act_optimizer.step()
        if self.count % 3000 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.count=0
        self.count+=1
    # 保存和读取历史网络参数
    def Save(self):
        torch.save(self.dqn.state_dict(), 'actor_critic_dqn_net_model.pth')
        torch.save(self.act.state_dict(), 'actor_critic_act_net_model.pth')
        print(f"Model saved")
    def Load(self):
        self.dqn.load_state_dict(torch.load('actor_critic_dqn_net_model.pth'))
        self.target_dqn.load_state_dict(torch.load('actor_critic_dqn_net_model.pth'))
        self.act.load_state_dict(torch.load('actor_critic_act_net_model.pth'))
        self.dqn.eval()
        self.act.eval()
        self.target_dqn.eval()
        print(f"Model loaded")

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
    while True:
        env.render()
        action=agent.decide(state.unsqueeze(0))
        action = torch.tensor(action).to(device)
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
        agent.replay_buffer.append([state,action,reward,next_state])
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
# agent.Load()

episode_rewards=[]
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    if (episode+1) % 50 ==0:
        agent.Save()
draw(episode_rewards)

agent.Save()

episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)
