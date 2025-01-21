import numpy as np
import gymnasium as gym
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm


def take_step(observation,action):
    if action==0:
        if observation-1==0:
            return observation-1,-1
        else:
            return observation-1,0
    else:
        if observation+1==20:
            return observation+1,1
        else:
            return observation+1,0

discounting_factor=1

def nsteptd(step_size,V,n):
    observation=10
    T=float('inf')
    time=0
    states = [observation]
    rewards = [0]
    while True:
        if time<T:
            if np.random.random()<0.5:
                action=1
            else:
                action=0
        new_state,rew=take_step(states[time],action)
        states.append(new_state)
        rewards.append(rew)
        if new_state==0 or new_state==20:
            T=time+1

        update_time=time-n+1
        if update_time>=0:
            gt=0
            for i in range(update_time+1,min(update_time+n,T)+1):
                gt+=np.power(discounting_factor,(i-update_time-1))*rewards[i]
            if update_time+n<T:
                gt=gt+np.power(discounting_factor,n)*V[states[update_time+n]]
            V[states[update_time]]=V[states[update_time]]+step_size*(gt-V[states[update_time]])
        if update_time==T-1:
            break
        time+=1

        
step_size=0.1
num_episodes=300
num_runs=1000
errors=0
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0
steps = np.power(2, np.arange(0, 2))

# all possible alphas
alphas = np.arange(0, 1.1, 0.1)

# each run has 10 episodes
episodes = 10

# perform 100 independent runs
runs = 100

# track the errors for each (step, alpha) combination
errors = np.zeros((len(steps), len(alphas)))
for run in tqdm(range(0, runs)):
    for step_ind, step in enumerate(steps):
        for alpha_ind, alpha in enumerate(alphas):
            # print('run:', run, 'step:', step, 'alpha:', alpha)
            value = np.zeros(21)
            for ep in range(0, episodes):
                nsteptd(alpha,value,step)
                # calculate the RMS error
                errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / 19)
# take average
errors /= episodes * runs

for i in range(0, len(steps)):
    plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
plt.xlabel('alpha')
plt.ylabel('RMS error')
plt.ylim([0.25, 0.55])
plt.legend()

plt.savefig('figure_7_3.png')
plt.close()
