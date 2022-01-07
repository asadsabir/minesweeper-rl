import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time
from game import *
start = time.time()

#variables below determine the grid to be trained on and 'moves' determines number of moves for model to train on
columns = 9
rows = 9
mines = 10
moves = 1000000




env = game(columns,rows,mines)

try:

    with open(f'PPO-minesweeper-r{rows}c{columns}m{mines}.csv') as file_name:
        past_progress = np.loadtxt(file_name, delimiter=",").tolist()
    model = PPO.load(f'PPO-minesweeper-r{rows}c{columns}m{mines}',env=env)
    model.learn(total_timesteps=moves)

except:
    past_progress = []
    model = PPO('MlpPolicy', env).learn(total_timesteps=moves)




fig, ax = plt.subplots()
total_progress = past_progress + Progress
print(len(past_progress))
print(len(Progress))
print(len(total_progress))
quiet_progress = []
count = 0
sum = 0
for i in range(len(total_progress)):
    sum+= total_progress[i]
    count+=1
    if count%1000 == 0 and count != 0:
        quiet_progress.append(sum/1000)
        sum = 0
    elif i == len(total_progress) - 1:
        quiet_progress.append(sum/(len(total_progress)%1000))
x = range(0,len(quiet_progress))
print(x)
ax.plot(x, quiet_progress)


model.save(f'PPO-minesweeper-r{rows}c{columns}m{mines}')
a = np.asarray(total_progress)
np.savetxt(f'PPO-minesweeper-r{rows}c{columns}m{mines}.csv',a,delimiter=',')


end = time.time()
seconds = end - start
print('runtime:',f'{seconds/60}minutes')
plt.show()
