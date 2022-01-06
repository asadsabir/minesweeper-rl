import random
import math
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym import Env

Progress = []

class tile:
    mine = False
    visible = False
    value = 0

class game(Env):
    def __init__(self, columns, rows, mines):
        obs = []
        for a in range(columns*rows):
            obs.append(11)
        self.metadata = None
        self.spec = None
        self.reward_range = (-100,73)
        self.observation_space = spaces.Box(low=0, high=11,shape=(rows,columns), dtype=np.int)
        self.action_space = spaces.Discrete(columns*rows)
        self.rows = rows
        self.columns = columns
        self.mines = mines

    def reset(self):
        self.grid = []
        self.mine_count = 0
        self.move_number = 0
        for j in range(self.rows):
            row = []
            visible_row = []
            for i in range(self.columns):
                visible_row.append(9)
                row.append(tile())
                
            self.grid.append(row)
        
        def pick_mines():
            chosen_row = random.randrange(self.rows)
            chosen_cell = random.randrange(self.columns)
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

        while self.mine_count < self.mines:
            pick_mines()
        
        self.visible_grid = []
        for a in range(self.columns*self.rows):
            self.visible_grid.append(9)
        
        return np.array(self.visible_grid).reshape(9,9)

    def step(self,action):

        def game_won():

            safe_cell_count = 0
            for row in self.grid:
                for cell in row:
                    if not cell.mine and cell.visible:
                        safe_cell_count +=1
            if safe_cell_count == self.rows*self.columns - self.mines:
                return True
            else:
                return False
        
        clicked = self.grid[math.floor(action/self.rows)][action%self.columns]
        if clicked.mine or clicked.visible:
            Progress.append(self.move_number)
            self.visible_grid[action] = 10
            if self.move_number == 1:
                return np.array(self.visible_grid).reshape(self.rows,self.columns), 0,True,{}
            return np.array(self.visible_grid).reshape(self.rows,self.columns), -100,True,{}
        else:
            self.move_number +=1
            clicked.visible = True
            self.visible_grid[action] = clicked.value
            if game_won():
                Progress.append(self.move_number)
            return np.array(self.visible_grid).reshape(self.rows,self.columns), self.move_number, game_won(),{}

    def render(self):
        pass

    def close(self):
        pass