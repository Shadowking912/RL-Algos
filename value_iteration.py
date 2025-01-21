import numpy as np
import gymnasium as gym
import sys

env = gym.make("CliffWalking-v0")
transitions=env.unwrapped.P
observation,info=env.reset()
x=list(transitions[37][2][0])
x[2]=10
transitions[35][2]=[tuple(x)]


V=np.zeros(env.observation_space.n)
discounting_factor=0.9
theta=1e-10
num_iter=0
while(True):
    delta=0
    for s in range(env.observation_space.n):
        v=V[s].copy()
        q=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            prob,next_state,rew,_=transitions[s][action][0]
            q[action]=prob*(rew+discounting_factor*V[next_state])
        V[s]=np.max(q)
        delta=max(delta,abs(v-V[s]))
    if delta<theta:
        break
    num_iter+=1

def display(v):
    grid=np.zeros((4,12))
    for i in range(len(v)):
        row=i//12
        column=i%12
        grid[row,column]=v[i]
    return grid

print("iterations: ",num_iter)
grid=display(V)
print(grid)
