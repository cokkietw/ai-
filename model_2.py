import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
class GymTaxi_env:
    def __init__(self):
        self.action_space=6
        self.observation_space=500
        self.mp=[]
        self.reset()
        self.state=0
    def reset(self):
        self.mp=[]
        self.mp.append(random.randint(1,5))#0 Taxi行
        self.mp.append(random.randint(1,5))#1 Taxi列
        self.mp.append(random.randint(0,3))#2 目的地 0R(1,1),1Y(5,1),2G(1,5),3B(5,4)
        self.mp.append(random.randint(0,3))#3 乘客位置 4在车上
        # print(self.mp)
        self.make_state()
    def make_state(self):
        self.state=(self.mp[0]-1)*100+(self.mp[1]-1)*20+self.mp[2]*5+self.mp[3]
    def render(self,second=0.5):# H乘客 M目的地 T出租车 
    ### 使用外部终端食用更佳
        mp=[['+','-','-','-','-','-','-','-','-','-','+'],
            ['|','R',':',' ','|',' ',':',' ',':','G','|'],
            ['|',' ',':',' ','|',' ',':',' ',':',' ','|'],
            ['|',' ',':',' ',':',' ',':',' ',':',' ','|'],
            ['|',' ','|',' ',':',' ','|',' ',':',' ','|'],
            ['|','Y','|',' ',':',' ','|','B',':',' ','|'],
            ['+','-','-','-','-','-','-','-','-','-','+']]
        if self.mp[3]!=4:
            if self.mp[3] == 0:
                mp[1][1]='H'
            elif self.mp[3] == 1:
                mp[5][1]='H'
            elif self.mp[3] == 2:
                mp[1][9]='H'
            elif self.mp[3] == 3:
                mp[5][7]='H'   
        if self.mp[2] == 0:
            mp[1][1]='M'
        elif self.mp[2] == 1:
            mp[5][1]='M'
        elif self.mp[2] == 2:
            mp[1][9]='M'
        elif self.mp[2] == 3:
            mp[5][7]='M'   
        mp[self.mp[0]][self.mp[1]*2-1]='T'
        for line in mp:
            for elem in line:
                print(elem,end='')
            print()
        time.sleep(second)
        for _ in range(7):
            sys.stdout.write(f'\033[{1}A')  # 移动光标向上 n 行
            sys.stdout.write(f'\033[2K' * 1)  # 清除光标当前行的内容
            sys.stdout.flush()
    def common_step(self,action):# state,reward,done
        mp=self.mp
        if self.mp[2] == self.mp[3]:
            return mp,40,True
        elif action == 0:# 上
            if self.mp[0] == 1:
                return mp,-1,False
            mp[0]-=1
            return mp,-1,False
        elif action == 1:# 下
            if self.mp[0] == 5:
                return mp,-1,False
            mp[0]+=1
            return mp,-1,False
        elif action == 2:# 左
            if self.mp[1] == 1 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==2 or self.mp[1]==4)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==3):
                return mp,-1,False
            mp[1]-=1
            return mp,-1,False
        elif action == 3:# 右
            if self.mp[1] == 5 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==1 or self.mp[1]==3)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==2):
                return mp,-1,False
            mp[1]+=1
            return mp,-1,False
        elif action == 4:# 接
            if (self.mp[3]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[3]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[3]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[3]==3 and self.mp[0]==5 and self.mp[1]==4):
                mp[3]=4
                return mp,20,False
            return mp,-10,False
        elif action == 5:# 放
            if self.mp[3]==4 and ((self.mp[2]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[2]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[2]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[2]==3 and self.mp[0]==5 and self.mp[1]==4)):
                return mp,20,True
            return mp,-10,False
    def only_Manhattan_step(self,action):# 曼哈顿距离修进reward 2目的地 3乘客
        mp=self.mp
        if self.mp[2] == self.mp[3]:
            return mp,40,True
        elif action == 0:# 上
            if self.mp[0] == 1:
                return mp,-50,False
            mp[0]-=1
            if mp[3]==0:
                return mp,10-(mp[0]+mp[1]-2),False
            elif mp[3]==1:
                return mp,(4-mp[0]+mp[1])-11,False
            elif mp[3]==2:
                return mp,10-(mp[0]+4-mp[1]),False
            elif mp[3]==3:
                return mp,(10-mp[0]-mp[1])-11,False
            elif mp[3]==4:
                if mp[2]==0:
                    return mp,10-(mp[0]+mp[1]-2),False
                elif mp[2]==1:
                    return mp,(4-mp[0]+mp[1])-11,False
                elif mp[2]==2:
                    return mp,10-(mp[0]+4-mp[1]),False
                elif mp[2]==3:
                    return mp,(10-mp[0]-mp[1])-11,False
        elif action == 1:# 下
            if self.mp[0] == 5:
                return mp,-50,False
            mp[0]+=1
            if mp[3]==0:
                return mp,(mp[0]+mp[1]-2)-11,False
            elif mp[3]==1:
                return mp,10-(4-mp[0]+mp[1]),False
            elif mp[3]==2:
                return mp,(mp[0]+4-mp[1])-11,False
            elif mp[3]==3:
                return mp,10-(10-mp[0]-mp[1]),False
            elif mp[3]==4:
                if mp[2]==0:
                    return mp,(mp[0]+mp[1]-2)-11,False
                elif mp[2]==1:
                    return mp,10-(4-mp[0]+mp[1]),False
                elif mp[2]==2:
                    return mp,(mp[0]+4-mp[1])-11,False
                elif mp[2]==3:
                    return mp,10-(10-mp[0]-mp[1]),False
        elif action == 2:# 左
            if self.mp[1] == 1 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==2 or self.mp[1]==4)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==3):
                return mp,-50,False
            mp[1]-=1
            if mp[3]==0:
                return mp,10-(mp[0]+mp[1]-2),False
            elif mp[3]==1:
                return mp,10-(4-mp[0]+mp[1]),False
            elif mp[3]==2:
                return mp,(mp[0]+4-mp[1])-11,False
            elif mp[3]==3:
                return mp,(10-mp[0]-mp[1])-11,False
            elif mp[3]==4:
                if mp[2]==0:
                    return mp,10-(mp[0]+mp[1]-2),False
                elif mp[2]==1:
                    return mp,10-(4-mp[0]+mp[1]),False
                elif mp[2]==2:
                    return mp,(mp[0]+4-mp[1])-11,False
                elif mp[2]==3:
                    return mp,(10-mp[0]-mp[1])-11,False
        elif action == 3:# 右
            if self.mp[1] == 5 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==1 or self.mp[1]==3)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==2):
                return mp,-50,False
            mp[1]+=1
            if mp[3]==0:
                return mp,(mp[0]+mp[1]-2)-11,False
            elif mp[3]==1:
                return mp,(4-mp[0]+mp[1])-11,False
            elif mp[3]==2:
                return mp,10-(mp[0]+4-mp[1]),False
            elif mp[3]==3:
                return mp,10-(10-mp[0]-mp[1]),False
            elif mp[3]==4:
                if mp[2]==0:
                    return mp,(mp[0]+mp[1]-2)-11,False
                elif mp[2]==1:
                    return mp,(4-mp[0]+mp[1])-11,False
                elif mp[2]==2:
                    return mp,10-(mp[0]+4-mp[1]),False
                elif mp[2]==3:
                    return mp,10-(10-mp[0]-mp[1]),False
        elif action == 4:# 接
            if (self.mp[3]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[3]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[3]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[3]==3 and self.mp[0]==5 and self.mp[1]==4):
                mp[3]=4
                return mp,20,False
            return mp,-50,False
        elif action == 5:# 放
            if self.mp[3]==4 and ((self.mp[2]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[2]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[2]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[2]==3 and self.mp[0]==5 and self.mp[1]==4)):
                return mp,20,True
            return mp,-50,False
    def advanded_Manhattan_step(self,action):# 加入中介点的曼哈顿距离修进reward 2目的地 3乘客
        mp=self.mp
        if self.mp[2] == self.mp[3]:
            return mp,40,True
        elif action == 0:# 上
            if self.mp[0] == 1:
                return mp,-50,False
            mp[0]-=1
            if mp[3]==0:
                if (mp[0]>3 and mp[1]>1):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,10-(abs(1-mp[0])+abs(1-mp[1])),False
            elif mp[3]==1:
                if (mp[0]>3 and mp[1]>1):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,(abs(5-mp[0])+abs(1-mp[1]))-11,False
            elif mp[3]==2:
                if (mp[0]>3 and mp[1]<4):
                    return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,10-(abs(1-mp[0])+abs(5-mp[1])),False
            elif mp[3]==3:
                if (mp[0]>3 and mp[1]<4):
                    return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
            elif mp[3]==4:
                if mp[2]==0:
                    if (mp[0]>3 and mp[1]>1):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,10-(abs(1-mp[0])+abs(1-mp[1])),False
                elif mp[2]==1:
                    if (mp[0]>3 and mp[1]>1):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,(abs(5-mp[0])+abs(1-mp[1]))-11,False
                elif mp[2]==2:
                    if (mp[0]>3 and mp[1]<4):
                        return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                    elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,10-(abs(1-mp[0])+abs(5-mp[1])),False
                elif mp[2]==3:
                    if (mp[0]>3 and mp[1]<4):
                        return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                    elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
        elif action == 1:# 下
            if self.mp[0] == 5:
                return mp,-50,False
            mp[0]+=1
            if mp[3]==0:
                if (mp[0]>3 and mp[1]>1):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,(abs(1-mp[0])+abs(1-mp[1]))-11,False
            elif mp[3]==1:
                if (mp[0]>3 and mp[1]>1):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,10-(abs(5-mp[0])+abs(1-mp[1])),False
            elif mp[3]==2:
                if (mp[0]>3 and mp[1]<4):
                    return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,(abs(1-mp[0])+abs(5-mp[1]))-11,False
            elif mp[3]==3:
                if (mp[0]>3 and mp[1]<4):
                    return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
            elif mp[3]==4:
                if mp[2]==0:
                    if (mp[0]>3 and mp[1]>1):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,(abs(1-mp[0])+abs(1-mp[1]))-11,False
                elif mp[2]==1:
                    if (mp[0]>3 and mp[1]>1):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    elif (mp[0]==3 and mp[1]>1)or(mp[1]>2):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,10-(abs(5-mp[0])+abs(1-mp[1])),False
                elif mp[2]==2:
                    if (mp[0]>3 and mp[1]<4):
                        return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                    elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,(abs(1-mp[0])+abs(5-mp[1]))-11,False
                elif mp[2]==3:
                    if (mp[0]>3 and mp[1]<4):
                        return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                    elif (mp[0]==3 and mp[1]<4)or(mp[1]<3):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
        elif action == 2:# 左
            if self.mp[1] == 1 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==2 or self.mp[1]==4)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==3):
                return mp,-50,False
            mp[1]-=1
            if mp[3]==0:
                if (mp[0]>3 or mp[1]>2):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,10-(abs(1-mp[0])+abs(1-mp[1])),False
            elif mp[3]==1:
                if (mp[0]>3 or mp[1]>2):
                    return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                else:
                    return mp,10-(abs(5-mp[0])+abs(1-mp[1])),False
            elif mp[3]==2:
                if (mp[1]<4):
                    return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                else:
                    return mp,(abs(1-mp[0])+abs(5-mp[1]))-11,False
            elif mp[3]==3:
                if (mp[1]<4):
                    return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                elif(mp[1]==4):
                    return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
                else:
                    return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
            elif mp[3]==4:
                if mp[2]==0:
                    if (mp[0]>3 or mp[1]>2):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,10-(abs(1-mp[0])+abs(1-mp[1])),False
                elif mp[2]==1:
                    if (mp[0]>3 or mp[1]>2):
                        return mp,6-(abs(3-mp[0])+abs(1-mp[1])),False
                    else:
                        return mp,10-(abs(5-mp[0])+abs(1-mp[1])),False
                elif mp[2]==2:
                    if (mp[1]<4):
                        return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                    else:
                        return mp,(abs(1-mp[0])+abs(5-mp[1]))-11,False
                elif mp[2]==3:
                    if (mp[1]<4):
                        return mp,(abs(3-mp[0])+abs(4-mp[1]))-7,False
                    elif(mp[1]==4):
                        return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
                    else:
                        return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
        elif action == 3:# 右
            if self.mp[1] == 5 or ((self.mp[0]==4 or self.mp[0]==5)and (self.mp[1]==1 or self.mp[1]==3)) or ((self.mp[0]==1 or self.mp[0]==2)and self.mp[1]==2):
                return mp,-50,False
            mp[1]+=1
            if mp[3]==0:
                if (mp[0]>3 or mp[1]>2):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,(abs(1-mp[0])+abs(1-mp[1]))-11,False
            elif mp[3]==1:
                if (mp[0]>3 or mp[1]>2):
                    return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                else:
                    return mp,(abs(5-mp[0])+abs(1-mp[1]))-11,False
            elif mp[3]==2:
                if (mp[1]<4):
                    return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                else:
                    return mp,10-(abs(1-mp[0])+abs(5-mp[1])),False
            elif mp[3]==3:
                if (mp[1]<4):
                    return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                elif(mp[1]==4):
                    return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
                else:
                    return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
            elif mp[3]==4:
                if mp[2]==0:
                    if (mp[0]>3 or mp[1]>2):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,(abs(1-mp[0])+abs(1-mp[1]))-11,False
                elif mp[2]==1:
                    if (mp[0]>3 or mp[1]>2):
                        return mp,(abs(3-mp[0])+abs(1-mp[1]))-7,False
                    else:
                        return mp,(abs(5-mp[0])+abs(1-mp[1]))-11,False
                elif mp[2]==2:
                    if (mp[1]<4):
                        return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                    else:
                        return mp,10-(abs(1-mp[0])+abs(5-mp[1])),False
                elif mp[2]==3:
                    if (mp[1]<4):
                        return mp,6-(abs(3-mp[0])+abs(4-mp[1])),False
                    elif(mp[1]==4):
                        return mp,10-(abs(5-mp[0])+abs(4-mp[1])),False
                    else:
                        return mp,(abs(5-mp[0])+abs(4-mp[1]))-11,False
        elif action == 4:# 接
            if (self.mp[3]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[3]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[3]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[3]==3 and self.mp[0]==5 and self.mp[1]==4):
                mp[3]=4
                return mp,20,False
            return mp,-50,False
        elif action == 5:# 放
            if self.mp[3]==4 and ((self.mp[2]==0 and self.mp[0]==1 and self.mp[1]==1) or (self.mp[2]==1 and self.mp[0]==5 and self.mp[1]==1) or (self.mp[2]==2 and self.mp[0]==1 and self.mp[1]==5) or (self.mp[2]==3 and self.mp[0]==5 and self.mp[1]==4)):
                return mp,20,True
            return mp,-50,False
# 生成环境
env = GymTaxi_env()

# Q-learning类
class Qlearning:
    def __init__(self,env,gamma=0.97,learning_rate=0.1,epsilon=0.01,train_episode=5000,test_episode=5000,max_step=100):
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.epsilon=epsilon
        self.train_episode=train_episode
        self.test_episode=test_episode
        self.action_n=env.action_space
        self.max_step=max_step
        # self.q=self.read()
        self.q=np.zeros((env.observation_space,env.action_space))
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
        np.savetxt('agent2.txt', self.q, fmt='%.3f')
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

# 训练函数
def train(agent,env):
    episode_reward=0
    env.reset()
    while True:
        action=agent.decide(env.state)
        next_state1,reward,done=env.advanded_Manhattan_step(action)
        episode_reward+=reward
        # print(episode_reward)
        next_state=(next_state1[0]-1)*100+(next_state1[1]-1)*20+next_state1[2]*5+next_state1[3]
        next_action=agent.decide(next_state)
        agent.learn(env.state,action,reward,next_state,next_action,done)
        # env.render(0.1)# 渲染效果可视化，注释掉可加快训练速度
        # agent.save()
        if done :#or step > agent.max_step:
            break
        env.mp,env.state,action=next_state1,next_state,next_action
    return episode_reward
# 测试函数
def test(agent,env):
    episode_reward=0
    env.reset()
    while True:
        action=agent.decide(env.state)
        next_state1,reward,done=env.advanded_Manhattan_step(action)
        episode_reward+=reward
        next_state=(next_state1[0]-1)*100+(next_state1[1]-1)*20+next_state1[2]*5+next_state1[3]
        next_action=agent.decide(next_state)
        env.render()# 渲染效果可视化，注释掉可加快测试速度
        if done:
            break
        env.mp,env.state,action=next_state1,next_state,next_action
    return episode_reward
# 绘制折线图
def draw(rewards):
    plt.clf()
    plt.plot([i for i in range(len(rewards))], rewards)
    s='episode:'+str(len(rewards))
    plt.xlabel(s)
    plt.title('result')
    plt.ylabel('reward')
    plt.pause(0.0001)
    plt.ioff()
    # plt.show()
# 训练
agent=Qlearning(env)
episode_rewards=[]
for episode in range(agent.train_episode):
    episode_reward=train(agent,env)
    print("train episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    draw(episode_rewards)
# 测试
episode_rewards=[]
for episode in range(agent.test_episode):
    episode_reward=test(agent,env)
    print("test episode:",episode+1,"reward:",episode_reward)
    episode_rewards.append(episode_reward)
    draw(episode_rewards)
# agent.save()