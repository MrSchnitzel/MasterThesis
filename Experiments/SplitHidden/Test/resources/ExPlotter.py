# Dopamine concentration:
# in:
#   time, concentrationR, concentrationL, Simulation_reset

# Weights:
#   time, range(1,175), Simulation_reset

# Performance:
#   time,angle to Target,Simulation_reset

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import csv
import numpy as np
import math
import Variables as Var


# style.use('seaborn-poster')
# style.use('ggplot')
style.use('seaborn-bright')
style.use('seaborn-whitegrid')
style.use('seaborn-paper')
import threading
import sys
import copy
import os
import time
class ExperimentPlotter:
    def __init__(self, animationInterval=0., xRange=0., xHard=False, plots=[], path=os.getcwd()):
        # self.animated = animationInterval > 0.
        self.animationInterval = float(animationInterval)
        self.xRange = int(xRange)
        self.hardLim =xHard
        self.plots = plots
        self.path = path
        self.name = os.path.basename(path)
        self.refreshLog = copy.deepcopy(ExperimentPlotter.refreshLog)
        self.animations = []
        for i in self.refreshLog:
            self.refreshData(i,True)
        # print(self.refreshLog)

    prefCommandMap = {
        "dir":1,
        "dirh":2,
        "a":3,
        "ba":4,
        "ar":5,
        "bar":6,
        "dist":7,
    }

    # prefCommandMap = {
    #     "dir": 1,
    #     "dirh": 1,
    #     "a": 2,
    #     "ba": 3,
    #     "ar": 4,
    #     "bar": 5,
    #     "dist": 6,
    # }
    def plot(self):
        #make figures for windows
        print('plotting')
        figs= []
        for window in self.plots:
            fig = plt.figure(self.name +' : ' + str(len(figs)))
            figs.append(fig)
            for rowID in range(len(window)):
                row = window[rowID]
                for axID in range(len(row)):
                    ax = row[axID]
                    arg = str(len(window)) + str(len(row)) + str(rowID * len(row) + axID + 1)
                    # print('sPlot: {}, arg: {}'.format(ax,arg))
                    plotAx = fig.add_subplot(int(arg))
                    title = str(ax)
                    # minRange = 0
                    for i in range(0, len(ax)):
                        command = ax[i]
                        t = self.commandsDic[command[0]]
                        f = t[1]
                        plotFunction = lambda t: f(self, t, plotAx, command)
                        if self.animated:
                            if command == ax[0]:
                                def addClear(t):
                                    print('clearing')
                                    plotAx.clear()
                                    plotFunction(t)
                                a = animation.FuncAnimation(fig, addClear, interval=self.animationInterval)
                            else:
                                a = animation.FuncAnimation(fig, plotFunction, interval=self.animationInterval)
                            self.animations.append(a)
                        else:
                            a = plotFunction(0)
                    # plotAx.set_title(title)
                    self.setRange(plotAx,ax)    
        # plt.show()

    def setRange(self,ax , commands):
        if self.rangeSet and not self.hardLim:
            minT = sys.maxint
            maxT = 0
            needsToSet = False
            for c in commands:
                dataBlob= self.commandsDic[c[0]][2]
                ar = self.refreshLog[dataBlob]['data']
                # print(ar)
                if dataBlob == 'state':
                    if len(ar) > self.xRange+1:
                        needsToSet = True
                        minT = min(minT, float(ar[-self.xRange][0]))
                        maxT = max(maxT, float(ar[-1][0]))
                else:
                    if len(ar[0]) > self.xRange+1:
                        needsToSet = True
                        minT = min(minT, float(ar[0][-self.xRange]))
                        maxT = max(maxT, float(ar[0][-1]))
            if needsToSet:
                ax.set_xlim(left=minT,right=maxT+0.01*maxT)

    animationInterval = -1
    @property
    def animated(self):
        return self.animationInterval > 0

    @property
    def rangeSet(self):
        return self.xRange > 0
    xRange = -1
    plots = []
    path = os.getcwd()
    @property
    def name(self):
        return os.path.basename(self.path)
    hardLim = False
    refreshLog = {
        "wbl":      {'tR':0., 'file':'left-weights', 'data':[]},
        "wbr":      {'tR':0., 'file':'right-weights', 'data':[]},
        "whl":      {'tR':0., 'file':'left-Head-weights', 'data':[]},
        "whr":      {'tR':0., 'file':'right-Head-weights', 'data':[]},
        "wx":       {'tR':0., 'file':'hidden-weights', 'data':[]},
        "perf":     {'tR':0., 'file':'performance', 'data':[]},
        "dope":     {'tR':0., 'file':'dope', 'data':[]},
        # "state":    {'tR':0., 'file':'state', 'data':[]},
    }


    commandsDic = {
        "wbl": ("Weights Body Left", lambda slf, t, ax, c: ExperimentPlotter.drawWeights(slf,t, ax, 'wbl', c),'wbl'),
        "wbr": ("Weights Body Right", lambda slf, t, ax, c: ExperimentPlotter.drawWeights(slf,t, ax, 'wbr', c),'wbr'),
        
        "whl": ("Weights Head Left", lambda slf, t, ax, c: ExperimentPlotter.drawWeights(slf,t, ax, 'whl', c),'whl'),
        "whr": ("Weights Head Right", lambda slf, t, ax, c: ExperimentPlotter.drawWeights(slf,t, ax, 'whr', c),'whr'),
        
        "wx": ("Weights hidden Layer", lambda slf, t, ax, c: ExperimentPlotter.drawWeights(slf,t, ax, 'wx', c),'wx'),

        "dir": ("direction", lambda slf, t, ax, c: ExperimentPlotter.plot_performance(slf,t, ax, 'perf', c),'perf'),
        "dirh": ("direction Head", lambda slf, t, ax, c: ExperimentPlotter.plot_performance(slf,t, ax, 'perf', c),'perf'),

        "a": ("angle", lambda slf, t, ax, c: ExperimentPlotter.plot_performance(slf,t, ax, 'perf', c),'perf'),
        "ar": ("raw angle", lambda slf, t, ax,c: ExperimentPlotter.plot_performance(slf,t, ax,'perf', c),'perf'),
        
        "ba": ("body angle", lambda slf, t, ax,c: ExperimentPlotter.plot_performance(slf,t, ax,'perf', c),'perf'),
        "bar": ("raw body angle", lambda slf, t, ax,c: ExperimentPlotter.plot_performance(slf,t, ax,'perf', c),'perf'),
        
        "dl": ("dope body left", lambda slf, t, ax, c: ExperimentPlotter.plot_dope(slf,t, ax, 'dope', c),'dope'),
        "dr": ("dope body right", lambda slf, t, ax, c: ExperimentPlotter.plot_dope(slf,t, ax,'dope', c),'dope'),
        
        "dhl": ("dope head left", lambda slf, t, ax,c: ExperimentPlotter.plot_dope(slf,t, ax,'dope', c),'dope'),
        "dhr": (" Dope Head Right", lambda slf, t, ax,c: ExperimentPlotter.plot_dope(slf,t, ax,'dope', c),'dope'),
        
        "dist": ("distance", lambda slf, t, ax,c: ExperimentPlotter.plot_performance(slf,t, ax,'perf', c),'perf'),
    }

    @staticmethod
    def calcPlots(string):
        windows = [[[[plot.split('+')
                      for plot in cell.split(' ')]
                     for cell in row.split('_')]
                    for row in window.split(',')]
                   for window in string.split(';')]
        return windows

    def askForPrintsAndLayout(self):
        for command in ExperimentPlotter.commandsDic:
            print("{}:\t{}".format(command, ExperimentPlotter.commandsDic[command][0]))
        print('+ => add option to plot')
        print('space seperation => same plot')
        print('underline seperation => additional plot same row')
        print('comma seperation => new row')
        print('semicolon seperation => additional window')
        input = raw_input("enter your choice ")
        return self.calcPlots(input)
    
    def refreshData(self, logId,force=False):
        logger = self.refreshLog[logId]
        matrix = []
        if force or (self.animated and logger["tR"] <= time.time() - self.animationInterval):
            with open(self.path+'/'+logger['file'] + '.csv') as csv_file:
                # with open('left-weights.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for i in csv_reader:
                    matrix.append(i)
                # print(matrix)
                if len(matrix) < 2:
                    return []
                if type(matrix[0][0]) is str:
                    matrix = matrix[1:]
                if not(logId == 'state'):
                    matrix = zip(*matrix)
                logger['tR'] = time.time()
                logger['data'] = matrix
                # print(len(matrix))
        else:
            matrix = logger['data']
        return self.corpRange(matrix,logId)

    def corpRange(self, matrix,logId):
        if not self.setRange:
            return matrix
        if not (logId == 'state'):
            if len(matrix) == 0:
                # print(matrix)
                return matrix
            if len(matrix[0]) < self.xRange:
                return matrix
            matrix = copy.copy(matrix)
            for i in  range(0,len(matrix)):
                matrix[i] = matrix[i][-self.xRange:]
        else:
            if len(matrix) < self.xRange:
                return matrix
            matrix = copy.copy(matrix)
            matrix = matrix[-self.xRange:]
        return matrix

    def drawWeights(self, t, ax, logId, command):
        matrix = self.refreshData(logId)

        # ax.clear()
        #Time Array
        tAr = matrix[0]
        id = 1
        for line in matrix[1:]:  # 0,120,20):
            mT = tAr

                
            if self.setRange and self.hardLim and len(line) > self.xRange:
                line = line[-self.xRange:]
                mT = tAr[-self.xRange:]
            
            
            setPixel = False
            if len(command) > 1:
                setPixel = command[1] == 'p'
            
            label = ''
            if setPixel:
                if id > Var.numBrainNeurons:
                    label = 'W {}'.format(id-Var.numBrainNeurons)
                else:
                    pixel = Var.idMapper(id)
                    label = 'Px({},{})'.format(pixel.x,pixel.y)
            else:
                label = 'W {}'.format(id)

            if float(mT[0]) is not 0.:
                offset = float(mT[0])
                mT = map(lambda t: float(t) - offset, mT)
            
            pLine = ax.plot(mT, line, label=label)
            # ax.annotate(label + ": {:.2f}".format(float(line[-1])), xy=(mT[-1], line[-1]), size=11, va='center')
            id =id+1

        # ax.set_title(self.commandsDic[command[0]])
        ax.set_ylabel('Weight')
        # ax.set_xlabel('Time')

    def plot_performance(self, t, ax, logId,commands):
        matrix = self.refreshData(logId)
        matrix = copy.copy(matrix)
        
        cToMap = {
            ("dir", 10),
            ("dirh", 10),
            ("dist", 10),
        }
        
        for c, factor in cToMap:
            # print(len(matrix))
            # print(factor)
            # print(c)
            # print(self.prefCommandMap[c])
            matrix[self.prefCommandMap[c]] = map(lambda v: float(v)*factor,matrix[self.prefCommandMap[c]])

        if float(matrix[0][0]) is not 0.:
            offset = float(matrix[0][0])
            matrix[0] = map(lambda t: float(t)-offset, matrix[0])
        for command in commands:
            ax.plot(matrix[0],matrix[self.prefCommandMap[command]],label=self.commandsDic[command][0])
        
        ax.set_ylabel('Angle / Distance (x10) / Direction (x10)')
        ax.legend()
    
    dopeCommandMap = {
        "dr": 1,
        "dl": 2,
        "dhr": 3,
        "dhl": 4,
    }

    def plot_dope(self,t ,ax,logId,commands):
        matrix = self.refreshData(logId)
        if float(matrix[0][0]) is not 0.:
            offset = float(matrix[0][0])
            matrix[0] = map(lambda t: float(t)-offset, matrix[0])
        for command in commands:
            ax.plot(matrix[0],matrix[self.dopeCommandMap[command]],label=self.commandsDic[command][0])
        ax.set_ylabel('Concentration')
        ax.legend()

args = sys.argv[1:]
animat = args[0] 
xrange = args[1] 
plot = args[2]
paths = args[3:-1]
xHard = args[-1]==1

print(args)
plot = ExperimentPlotter.calcPlots(args[2])
print(plot)
threads = []
for path in paths:
    ex = ExperimentPlotter(animat, xrange, xHard, plot, path)
    ex.plot()
    # x = threading.Thread(target=ex.plot)
    # threads.append(x)
print('showing')
plt.show()
# for t in threads:
#     t.start()
# for t in threads:
#     t.join()
