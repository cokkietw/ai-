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
import copy
from collections import deque
env = gym.make('Pong-v4')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_reward_to_go(reward_list, gamma=0.99):
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # 递推衰减
        reward_arr[i] += gamma * reward_arr[i + 1]
    # 标准化奖励
    reward_arr =reward_arr- np.mean(reward_arr)
    if np.std(reward_arr):
        reward_arr /= np.std(reward_arr)
    return reward_arr

# 测试函数
def test(env):
    # env.render()
    episode_reward=0
    env.reset()
    N_step = 50
    N_sample = 10
    while True:
        env.render()
        max_rewards = -100
        max_actions = 0
        action_lists = []
        reward_lists = []
        for _ in range(N_sample):
            action_list = []
            for _ in range(N_step):
                now_action = random.randint(0,5)
                action_list.append(now_action)
            action_lists.append(action_list)
        for i in range(N_sample):
            now_env = copy.deepcopy(env)
            action_list = action_lists[i]
            reward_list = []
            for j in range(N_sample):
                # now_env.render()
                _, now_reward, now_terminal,_ = now_env.step(action_list[j])
                reward_list.append(now_reward)
                if now_terminal == True:
                    break
            reward_lists.append(calc_reward_to_go(reward_list).sum())
        for i in range(N_sample):
            if reward_lists[i] > max_rewards:
                max_rewards = reward_lists[i]
                max_actions = action_lists[i][0]
        _, reward, done,_ = env.step(action_lists[max_actions][0])
        episode_reward+=reward
        if done:
            break
    episode_reward = int(episode_reward)
    return episode_reward

# 绘制折线图
def draw(rewards):
    plt.plot([i for i in range(len(rewards))], rewards)
    plt.xlabel('episode')
    plt.title('result')
    plt.ylabel('reward')
    plt.show()

episode_rewards=[]
for episode in range(500):
    episode_reward=test(env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
draw(episode_rewards)

