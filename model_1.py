import gym
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# 生成环境
env = gym.make("Taxi-v2")
# 重置和渲染环境
obs = env.reset()
# env.render()
m = env.observation_space.n # 状态state
n = env.action_space.n  # 动作action
# Q-learning类
class Qlearning:
    def __init__(self,env,gamma=0.97,learning_rate=0.1,epsilon=0.01,train_episode=5000,test_episode=5000):
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.epsilon=epsilon
        self.train_episode=train_episode
        self.test_episode=test_episode
        self.action_n=env.action_space.n
        # self.q=self.read()
        self.q=np.zeros((env.observation_space.n,env.action_space.n))
    def decide(self,state):
        if random.random() > self.epsilon:
            action=self.q[state].argmax()
        else:
            action=np.random.randint(self.action_n)
        return action
    def learn(self,state,action,reward,next_state,next_action,done):
        action=int(action)
        next_action=int(next_action)
        loss=reward+self.gamma*self.q[next_state].max()-self.q[state,action]
        self.q[state,action]+=self.learning_rate*loss
    def save(self):
        np.savetxt('agent.txt', self.q, fmt='%.3f')
    def read(self):
        f = open("agent.txt")
        line = f.readline()
        data_list = []
        while line:
            num = list(map(float,line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
        data_array = np.array(data_list)
        return data_array

# Sarsa类

# 构造reward表
# def make_reward(env,reward_table):
#     num_states = env.observation_space.n
#     num_actions = env.action_space.n
#     reward_matrix = np.zeros((num_states, num_actions))

#     goal_state = 0  
#     for state in range(num_states):
#         for action in range(num_actions):
#             if action == 'dropoff' and state == goal_state:
#                 reward_matrix[state, action] = 20  # 成功送达的奖励
#             # elif state == 
#             elif action == 'pickup' and state == goal_state:
#                 reward_matrix[state, action] = -10  # 错误操作的惩罚
#             else:
#                 reward_matrix[state, action] = -1  # 每一步的惩罚

# 训练函数 关闭渲染可以加快训练速度
def train(agent,env):
    # env.render()
    episode_reward=0
    state=env.reset()
    while True:
        #env.render()
        action=agent.decide(state)
        next_state,reward,done,_=env.step(action)
        episode_reward+=reward
        next_action=agent.decide(next_state)
        if done:
            break
        agent.learn(state,action,reward,next_state,next_action,done)
        #agent.save()
        
        state,action=next_state,next_action
    return episode_reward
# 测试函数
def test(agent,env):
    episode_reward=0
    state=env.reset()
    env.render()
    while True:
        env.render()
        action=agent.decide(state)
        next_state,reward,done,_=env.step(action)
        episode_reward+=reward
        next_action=agent.decide(next_state)
        if done:
            break
        state,action=next_state,next_action
    return episode_reward
# 绘制折线图
def draw(x,rewards):
    plt.clf()
    plt.plot(x, rewards)
    s='episode:'+str(len(rewards))
    plt.xlabel(s)
    plt.title('result')
    plt.ylabel('reward')
    plt.pause(0.000001)
    plt.ioff()
    # plt.show()
# 训练
agent=Qlearning(env)
episode_rewards=[]
x=[]
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    x.append(episode)
    draw(x,episode_rewards)
# 测试
episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)
# agent.save()