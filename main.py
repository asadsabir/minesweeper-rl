import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from game import *
from stable_baselines3 import PPO

fig, ax = plt.subplots()
ln, = plt.plot([1], [1], 'ro')

columns = 9
rows = 9
mines = 10      
moves = 2e4     # <--- number of moves model trains for each point 
points = int(1e5)     # <--- number of points on graph


env = game(columns,rows,mines)
filep = f'PPO-minesweeper-r{rows}c{columns}m{mines}n'
model = PPO.load(filep,env=env)

global counter  
counter = 0
xdata, ydata = [], []
ymin = 12
ymax = 14

def init():
    ax.set_ylabel('rolling average number of moves per game')
    ax.set_xlabel('moves played in 1000s')
    ax.set_xlim(moves/1000, moves*2/1000)
    ax.set_ylim(ymin, ymax)
    return ln,ax

def reset_counter():
    global counter
    counter = len(current_progress)

def update(frame):

    global current_progress
    reset_counter()

    model.learn(total_timesteps=moves)

    average = np.sum(current_progress[counter:])/len(current_progress[counter:])

    xdata.append(frame/1000)
    ydata.append(average)

    if average > ymax:
        ax.set_ylim(ymin,average+0.1)
    elif average < ymin:
        ax.set_ylim(average-0.1,ymax)
    if frame > moves:
        ax.set_xlim(moves/1000,frame/1000)

    ln.set_data(xdata, ydata)
    
    if frame == moves*points:
        plt.close()

    return ln,ax

ani = FuncAnimation(fig, 
                    update, 
                    frames=np.linspace(moves,moves*points,points),
                    init_func=init, 
                    blit=False, 
                    interval = 2000) #interval is time (in ms) between calling update/rendering points, you can close graph during this time
plt.show()

model.save(filep)

with open(f'{filep}.csv') as file_name:
    total_progress = np.loadtxt(file_name, delimiter=",").tolist() + Progress
    a = np.asarray(total_progress)
    np.savetxt(f'{filep}.csv',a,delimiter=',')
    print('model saved')
    ra = 1e4    #number of games for rolling average to average over for final graph
    quiet_progress = []
    count = 0
    sum = 0
    for i in range(len(total_progress)):
        sum+= total_progress[i]
        count+=1
        if count%ra == 0 and count != 0:
            quiet_progress.append(sum/ra)
            sum = 0
        elif i == len(total_progress) - 1:
            quiet_progress.append(sum/(len(total_progress)%ra))

    plt.plot(quiet_progress)
    plt.xlabel(f'number of games played in {int(ra)}s')
    plt.ylabel('rolling average number of moves per game')
    plt.show()

    




