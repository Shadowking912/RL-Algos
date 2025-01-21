import numpy as np
import gymnasium as gym
import sys

env = gym.make("CliffWalking-v0")
transitions=env.unwrapped.P
observation,info=env.reset()
x=list(transitions[35][2][0])
x[2]=10
transitions[35][2]=[tuple(x)]

def policy_eval(V,pi,discounting_factor=0.9,theta=1e-10):
    while(True):
        delta=0
        for s in range(env.observation_space.n):
            v=V[s].copy()
            action=pi[s]
            prob,next_state,rew,_=transitions[s][action][0]
            V[s]= prob*(rew+discounting_factor*V[next_state])
            delta=max(delta,abs(v-V[s]))
        if(delta<theta):
            break
    
def policy_improvement(V,pi,discounting_factor=0.9):
    for s in range(env.observation_space.n):
        q=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            prob,next_state,rew,_=transitions[s][action][0]
            # print(rew)
            q[action]=prob*(rew+discounting_factor*V[next_state])
        pi[s]=np.argmax(q)


print("observation space:", env.observation_space)
print("action space:", env.action_space)
V=np.zeros(env.observation_space.n)
pi=np.zeros(env.observation_space.n)
pi_new=pi

def display(v):
    grid=np.zeros((4,12))
    for i in range(len(v)):
        row=i//12
        column=i%12
        grid[row,column]=v[i]
    return grid

threshold=0.001
num_iter=0
while(True):
    policy_eval(V,pi)
    old_policy=pi.copy()
    policy_improvement(V,pi)
    if np.array_equal(old_policy,pi):
        break
    pi=pi_new
    num_iter+=1

print("iterations: ",num_iter)
grid=display(V)
print(grid)
grid=display(pi)
print(grid)
