# 使用演员评论家网络，一局一局游戏传入，基于概率选择动作，加入值函数和奖励衰减机制处理奖励，TD更新值函数，最大化概率*奖励以更新策略网络
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
    image = image.unsqueeze(0)
    return image

env = gym.make('Pong-v4')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# V网络
class V_Net(nn.Module):
    def __init__(self):
        super(V_Net,self).__init__()
        # 卷积层V
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
        # 状态传播
        x = x.to(device)
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
        output = self.linear3(x)
        return output
# 策略网络
class p_Net(nn.Module):
    def __init__(self, n_actions=6):
        super(p_Net,self).__init__()
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
        x = x.to(device)
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
# 经验池
class ReplayMemory:
    def __init__(self,batch_size = 64):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.BATCH_SIZE = batch_size
    def add(self,state,action,reward,value,done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    def sample(self):
        num_state = len(self.states)
        batch_start_points = np.arange(0,num_state,self.BATCH_SIZE)
        memory_indicies = np.arange(num_state,dtype=int)
        np.random.shuffle(memory_indicies)
        batches = [memory_indicies[i:i+self.BATCH_SIZE] for i in batch_start_points]
        return self.states,self.actions,self.rewards,self.values,self.dones,batches
    def clear_memo(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
# 智能体
class Breakout_agent:
    def __init__(self,env,action_n=6,gamma=0.99, batch_size=64,epsilon = 0.01,
                 learning_rate=0.001,train_episode=5000,test_episode=5000,
                 replay_batch_size=64,lr_episode = 100,Lambda = 0.95,
                 EPSILON_CLIP = 0.2):
        # 智能体参数
        self.learning_rate=learning_rate
        self.train_episode=train_episode
        self.test_episode=test_episode
        self.action_n=action_n
        self.batch_size = batch_size
        self.gamma=gamma
        self.count=0
        self.epsilon = epsilon
        self.replay_batch_size=replay_batch_size
        self.lr_episode = lr_episode
        self.Lambda = Lambda
        self.EPSILON_CLIP = EPSILON_CLIP
        # 智能体网络
        self.Critic = V_Net().to(device)
        self.act=p_Net(action_n).to(device)
        self.target_act=p_Net(action_n).to(device)
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=self.learning_rate)

        # 经验池
        self.replay_buffer=ReplayMemory(self.replay_batch_size)
    # 基于概率选取动作
    def decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.act.forward(state)
        probs=probs.squeeze(0)
        value = self.Critic.forward(state)
        probs = probs.cpu().numpy()
        return np.random.choice(len(probs),1,p=probs),value
    # 测试时，选取最大概率的动作
    def test_decide(self,state):
        state=state.to(device)
        with torch.no_grad():
            probs=self.act.forward(state)
        probs=probs.squeeze(0)
        probs = probs.cpu().numpy()
        return probs.argmax().numpy() 
    # 学习函数
    def learn(self):
        self.target_act.load_state_dict(self.act.state_dict())
        memo_states,memo_actions,memo_rewards,memo_values,memo_dones,batches = self.replay_buffer.sample()
        T = len(memo_rewards)
        memo_advantages = []
        for t in range(T-1):
            discount = 1
            a_t = 0
            for k in range(t,T-1):
                a_t += memo_rewards[k] + self.gamma * memo_values[k+1] - memo_values[k]
                discount*=self.gamma*self.Lambda
            memo_advantages.append( torch.Tensor(a_t * discount))
        memo_advantages.append( torch.tensor(memo_rewards[T-1]).unsqueeze(0).unsqueeze(0).to(device))
        memo_states = torch.tensor([item.cpu().detach().numpy() for item in memo_states]).to(device)
        memo_actions = torch.tensor(memo_actions).to(device)
        memo_advantages= torch.tensor([item.cpu().detach().numpy() for item in memo_advantages]).to(device)
        memo_values = torch.tensor([item.cpu().detach().numpy() for item in memo_values]).to(device)
        for batch in batches:
            with torch.no_grad():
                old_log_prob = torch.distributions.Categorical(self.target_act.forward(memo_states[batch.tolist()])).log_prob(memo_actions[batch.tolist()].T)
            log_prob = torch.distributions.Categorical(self.act.forward(memo_states[batch.tolist()])).log_prob(memo_actions[batch.tolist()].T)
            # 重要性采样
            ratio = torch.exp(log_prob - old_log_prob).T
            # 更新限制
            limit1 = ratio * memo_advantages[batch.tolist()].squeeze(1)
            limit2 = torch.clamp(ratio,1-self.EPSILON_CLIP,1+self.EPSILON_CLIP)*memo_advantages[batch.tolist()].squeeze(1)
            
            act_loss = - torch.min(limit1,limit2).mean()
            
            new_Value = ((memo_advantages[batch.tolist()] + memo_values[batch.tolist()])).squeeze(1)
            old_Value = self.Critic.forward(memo_states[batch.tolist()])
            critic_loss = nn.MSELoss()(old_Value, new_Value)

            self.act_optimizer.zero_grad()
            act_loss.backward()
            self.act_optimizer.step()

            self.Critic_optimizer.zero_grad()
            critic_loss.backward()
            self.Critic_optimizer.step()

    # 保存和读取历史网络参数
    def Save(self):
        torch.save(self.target_act.state_dict(), 'PPO_act_net_model.pth')
        torch.save(self.Critic.state_dict(), 'PPO_Critic_net_model.pth')
        print(f"Model saved")
    def Load(self):
        self.act.load_state_dict(torch.load('PPO_act_net_model.pth'))
        self.Critic.load_state_dict(torch.load('PPO_Critic_net_model.pth'))
        self.act.eval()
        self.Critic.eval()
        print(f"Model loaded")



# 训练函数
def train(agent,env):
    episode_reward=0
    state=env.reset()
    state=preprocess(state)
    step=0
    while True:
        env.render()
        action , value=agent.decide(state.unsqueeze(0))
        next_state, reward, done, info = env.step(action)
        next_state=preprocess(next_state)
        episode_reward+=reward
        agent.replay_buffer.add(state,action,reward,value,done)
        step+=1
        if (step+1)%128 == 0 or done:
            agent.learn()
        if done:
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
        action,_=agent.test_decide(state.unsqueeze(0))
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
Max_reward = -100
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    agent.replay_buffer.clear_memo()
    if episode_reward > Max_reward:
        agent.Save()
        Max_reward = episode_reward
draw(episode_rewards)

episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)
