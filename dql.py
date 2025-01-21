import numpy as np
import gymnasium as gym
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm




actions={1:[i for i in range(10)],2:[0,1]}


def epsilon_greedy(epsilon,q1,observation):
    if np.random.random()<epsilon:
        action=np.random.choice(actions[observation])
    else:
        indices = np.where(q1[observation]==np.max(q1[observation]))[0]
        action=np.random.choice(indices)
    return action

def epsilon_greedy2(epsilon,q1,q2,observation):
    if np.random.random()<epsilon:
        action=np.random.choice(actions[observation])
    else:
        indices = np.where((q1[observation]+q2[observation])==np.max(q1[observation]+q2[observation]))[0]
        action=np.random.choice(indices)
    return action

def step(observation,action):
    if observation==1:
        return 0,np.random.normal(-0.1,1)
    elif observation==2 and action==0:
        return 1,0
    elif observation==2 and action==1:
        return 3,0

discounting_factor=1

def q_learning(step_size,Q,epsilon):
    observation=2
    left=0
    while True:
        action=epsilon_greedy(epsilon,Q,observation)
        if observation==2 and action==0:
            left+=1
        new_state,rew=step(observation,action)
        Q[observation][action]=Q[observation][action]+step_size*(rew+discounting_factor*np.max(Q[new_state])-Q[observation][action])
        observation=new_state
        if observation==0 or observation==3:
            break
    return left

def double_q(step_size,q1,q2,epsilon):
    observation=2
    left=0
    while True:
        action=epsilon_greedy2(epsilon,q1,q2,observation)
        if observation==2 and action==0:
            left+=1
        new_state,rew=step(observation,action)
        if np.random.random()<0.5:
            q1[observation][action]=q1[observation][action]+step_size*(rew+discounting_factor*q2[new_state][np.argmax(q1[new_state])]-q1[observation][action])
        else:
            q2[observation][action]=q2[observation][action]+step_size*(rew+discounting_factor*q1[new_state][np.argmax(q2[new_state])]-q2[observation][action])
        observation=new_state
        if observation==0 or observation==3:
            break
    return left
        
step_size=0.1
num_episodes=300
num_runs=1000
left_q=np.zeros((num_runs,num_episodes))
left_dq=np.zeros((num_runs,num_episodes))
for j in tqdm(range(num_runs),total=num_runs):
    Q=[np.zeros(2,dtype=np.float32),np.zeros(10,dtype=np.float32),np.zeros(2,dtype=np.float32),np.zeros(2,dtype=np.float32)]
    q1=[np.zeros(2,dtype=np.float32),np.zeros(10,dtype=np.float32),np.zeros(2,dtype=np.float32),np.zeros(2,dtype=np.float32)]
    q2=[np.zeros(2,dtype=np.float32),np.zeros(10,dtype=np.float32),np.zeros(2,dtype=np.float32),np.zeros(2,dtype=np.float32)]
    for i in range(num_episodes):
        left_q[j,i]=q_learning(step_size,Q,0.1)  
        left_dq[j,i]=double_q(step_size,q1,q2,0.1)

left_q=left_q.mean(axis=0)
left_dq=left_dq.mean(axis=0)
plt.plot(left_q,label="q")
plt.plot(left_dq,label="doubleq")
plt.legend()
plt.show()
