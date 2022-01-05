import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from gym import Env
from stable_baselines3 import PPO
import time
start = time.time()

class tile:
    mine = False
    visible = False
    value = 0

Progress = []

fig, ax = plt.subplots()

class game(Env):
    def __init__(self):
        obs = []
        for a in range(81):
            obs.append(11)
        self.metadata = None
        self.spec = None
        self.reward_range = (-100,73)
        self.observation_space = spaces.Box(low=0, high=11,shape=(9,9), dtype=np.int)
        self.action_space = spaces.Discrete(81)

    def reset(self):
        self.grid = []
        self.mine_count = 0
        self.move_number = 1
        for j in range(9):
            row = []
            visible_row = []
            for i in range(9):
                visible_row.append(9)
                row.append(tile())
                
            self.grid.append(row)
        
        def pick_mines():
            chosen_row = random.randrange(9)
            chosen_cell = random.randrange(9)
            if self.grid[chosen_row][chosen_cell].mine == True:
                pick_mines()
            else:
                self.grid[chosen_row][chosen_cell].mine = True

                def mine_values(i,j):
                    try:
                        self.grid[chosen_row+i][chosen_cell+j].value +=1
                    except:
                        #incase list is out of range
                        pass
                
                mine_values(1,1)
                mine_values(-1,-1)
                mine_values(-1,1)
                mine_values(1,-1)
                mine_values(0,1)
                mine_values(0,-1)
                mine_values(1,0)
                mine_values(-1,0)

                self.mine_count = self.mine_count + 1 

        while self.mine_count < 10:
            pick_mines()
        
        self.visible_grid = []
        for a in range(81):
            self.visible_grid.append(9)
        
        return np.array(self.visible_grid).reshape(9,9)

    def step(self,action):

        def game_won():

            safe_cell_count = 0
            for row in self.grid:
                for cell in row:
                    if not cell.mine and cell.visible:
                        safe_cell_count +=1
            if safe_cell_count == 71:
                return True
            else:
                return False
        
        clicked = self.grid[math.floor(action/9)][action%9]
        if clicked.mine or clicked.visible:
            Progress.append(self.move_number)
            self.visible_grid[action] = 10
            if self.move_number == 1:
                return np.array(self.visible_grid).reshape(9,9), 0,True,{}
            return np.array(self.visible_grid).reshape(9,9), -100,True,{}
        else:
            self.move_number +=1
            clicked.visible = True
            self.visible_grid[action] = clicked.value
            if game_won():
                Progress.append(self.move_number)
            return np.array(self.visible_grid).reshape(9,9), self.move_number, game_won(),{}

    def render(self):
        pass

    def close(self):
        pass

env = game()

model = PPO.load('PPO-minesweeper',env=game())



model.learn(total_timesteps=1000000)

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

# def animate(i):
    
#     try:
#         global x
#         x = np.append(x,i)
#         ax.clear()
#         ax.plot(x,Progress[x])  # update the data.
#     except:
#         pass

# ani = animation.FuncAnimation(fig, animate)
model.save('PPO-minesweeper')
end = time.time()
seconds = end - start
print('runtime:',seconds)
plt.show()
