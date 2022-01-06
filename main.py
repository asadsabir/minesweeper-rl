import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from gym import Env
from stable_baselines3 import PPO
import time
from game import *
start = time.time()



fig, ax = plt.subplots()



env = game(9,9,10)

model = PPO.load('PPO-minesweeper',env=env)



model.learn(total_timesteps=100000)

quiet_progress = []
count = 0
sum = 0
for i in range(len(Progress)):
    sum+= Progress[i]
    count+=1
    if count%1000 == 0 and count != 0:
        quiet_progress.append(sum/1000)
        sum = 0
    elif i == len(Progress) - 1:
        quiet_progress.append(sum/(len(Progress)%1000))
x = range(0,len(quiet_progress))
print(x)
ax.plot(x, quiet_progress)

model.save('PPO-minesweeper')
end = time.time()
seconds = end - start
print('runtime:',seconds)
plt.show()
