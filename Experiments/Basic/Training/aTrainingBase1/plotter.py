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

AnimInterv = 5000
xRange = 1000

setXLim = False  # True
animated = False  # False

for i in range(Var.numBrainNeurons):
    p = Var.idMapper(i+1)
    print('i: {} corresponds to Pixel({},{}) '.format(i, p.x, p.y))


def drawWeights(i, ax, name):
    # print(i,ax,name)
    # coordinates = [[(float(i),j+((2.*i)/(j+1.)))for i in range(NENTRIES)]for j in range(NLINES)]
    with open(name + '.csv') as csv_file:
        # with open('left-weights.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        matrix = []
        for i in csv_reader:
            matrix.append(i)
        # print(matrix)
        NLINES = len(matrix[1])
        NENTRIES = len(matrix)

        # NLINES = 17
        coordinates = [[(matrix[i][0], matrix[i][j])
                        for i in range(1, NENTRIES)]for j in range(1, NLINES)]
        # print(coordinates[0])

    # bins = np.arange(0,40,5)
    ax.clear()
    xLeft = 0.
    xRight = 0.
    for i in range(len(coordinates)):  # 0,120,20):
        line = coordinates[i]
        xA = []
        yA = []
        for (x, y) in line:
            xA.append(x)
            yA.append(float(y))
        # deviation = np.std(yA)

        # print('i: {} corresponds to Pixel({},{}) '.format(i, NUM_PIXEL_X -
        #                                                   (i_invers % (NUM_PIXEL_Y)), int(math.floor(i_invers/NUM_PIXEL_Y))))

        # if deviation>1:
        if i > 9:
            pLine = ax.plot(xA, yA, label='jointNeuron({})'.format(i))
            ax.annotate('jointNeuron({}): {:.2f} '.format(i, yA[len(
                yA)-1]), xy=(xA[len(xA)-1], yA[len(yA)-1]), size=11, va='center')

        else:
            p = Var.idMapper(i+1)
            pLine = ax.plot(xA, yA, label='Pixel({},{}) '.format(p.x, p.y))
            # if deviation > 3:

            ax.annotate('Pixel({},{}): {:.2f} '.format(p.x, p.y, yA[len(
                yA)-1]), xy=(xA[len(xA)-1], yA[len(yA)-1]), size=11, va='center')

        # plt.hist(yA,bins,histtype='bar',rwidth=0.3)
        if i == 0:
            if len(xA) > xRange:
                xLeft = xA[-xRange]
            xRight = xA[-1]

    ax.set_title(name[:-8])
    # ax.set_xlabel('Time')
    ax.set_ylabel('Weight')

    if setXLim and xLeft > 0.:
        ax.set_xlim(left=float(xLeft), right=float(xRight))

    # ax.title(name)


def plot_dope(i, ax):
    with open('dope.csv') as csv_file:
        # with open('left-weights.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # matrix = []
        tAr = []
        rAr = []
        lAr = []
        rHAr = []
        lHAr = []
        doIt = 1
        for i in csv_reader:
            if doIt > 0:
                doIt = doIt-1
            else:
                t, r, l, rh, lh = i
                tAr.append(t)
                rAr.append(r)
                lAr.append(l)
                rHAr.append(rh)
                lHAr.append(lh)
        ax.clear()
        rightdope = ax.plot(tAr, rAr, label='right Dope lvl')
        leftdope = ax.plot(tAr, lAr, label='left Dope lvl')

        rightHeaddope = ax.plot(tAr, rHAr, label='right Head Dope lvl')
        leftHeaddope = ax.plot(tAr, lHAr, label='left Head Dope lvl')
        # ax.set_xlabel('Time')
        if setXLim and len(tAr) > xRange:
            ax.set_xlim(left=float(tAr[-xRange]), right=float(tAr[-1]))
        ax.set_ylabel('concentration')
        ax.set_title('Dope')
        ax.legend()


def plot_angles(i, ax):
    with open('performance.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tAr = []
        angleAr = []
        rawAngleAr = []
        distanceAr = []
        rawBodyAngleAr = []
        bodyAngleAr = []
        csv_reader.next()
        for i in csv_reader:
            t, dire, direH, a, ba, ra, rba, dist = i
            tAr.append(t)
            angleAr.append(a)
            rawAngleAr.append(ra)
            rawBodyAngleAr.append(rba)
            distanceAr.append(float(dist) * 10)
            bodyAngleAr.append(ba)
        ax.clear()
        angle = ax.plot(tAr, angleAr, label='angle')
        bodyAngle = ax.plot(tAr, bodyAngleAr, label='body angle')
        # rawAngle = ax.plot(tAr, rawAngleAr, label='raw angle')
        # rawBodyAngle = ax.plot(tAr, rawBodyAngleAr, label='raw body angle')
        distance = ax.plot(tAr, distanceAr, label='distance')
        # ax.set_xlabel('Time')
        ax.set_ylabel('angle/distance x10/direction x10')
        ax.set_title('Performance')
        ax.legend()
        if setXLim and len(tAr) > xRange:
            ax.set_xlim(left=float(tAr[-xRange]), right=float(tAr[-1]))


def plot_performance(i, ax):
    with open('performance.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tAr = []
        angleAr = []
        distanceAr = []
        direcAr = []
        direcHAr = []
        bodyAngleAr = []
        rawBodyAngleAr = []
        rawAngleAr = []
        csv_reader.next()
        for i in csv_reader:
            t, dire, direH, a, ba, ra, rba, dist = i
            tAr.append(t)
            direcAr.append((float(dire) * 30))
            direcHAr.append((float(direH) * 30))
            # angleAr.append(a)
            rawAngleAr.append(ra)
            # distanceAr.append(float(dist) * 10)
            bodyAngleAr.append(ba)
            rawBodyAngleAr.append(rba)
        ax.clear()
        direction = ax.plot(tAr, direcAr, label='direction')
        directionH = ax.plot(tAr, direcHAr, label='direction Head')
        # angle = ax.plot(tAr, angleAr, label='angle')
        rawAngle = ax.plot(tAr, rawAngleAr, label='raw angle')
        bodyAngle = ax.plot(tAr, bodyAngleAr, label='body angle')
        rawBodyAngle = ax.plot(tAr, rawBodyAngleAr, label='raw body angle')
        # distance = ax.plot(tAr, distanceAr, label='distance')
        # ax.set_xlabel('Time')
        ax.set_ylabel('angle/distance x10/direction x10')
        ax.set_title('Performance')
        ax.legend()
        if setXLim and len(tAr) > xRange:
            ax.set_xlim(left=float(tAr[-xRange]), right=float(tAr[-1]))


fig = plt.figure()  # performance
fig2 = plt.figure()  # head
fig3 = plt.figure()  # body
fig4 = plt.figure()  # hidden

axLeft = fig3.add_subplot(211)
axRight = fig3.add_subplot(212)

axLeftH = fig2.add_subplot(211)
axRightH = fig2.add_subplot(212)

axPerf = fig.add_subplot(211)
axAngle = fig.add_subplot(212)
# axDope2 = fig.add_subplot(313)

axHidden = fig4.add_subplot(111)

if animated:
    # axLeft = fig.add_subplot(421)
    # axRight = fig.add_subplot(422)
    # axDope = fig.add_subplot(423)
    # axDope2 = fig.add_subplot(424)
    # axAngle = fig.add_subplot(413)
    # axPerf = fig.add_subplot(414)

    a1 = animation.FuncAnimation(fig3, drawWeights, fargs=(
        axLeft, 'left-weights'), interval=AnimInterv)
    a2 = animation.FuncAnimation(fig3, drawWeights, fargs=(
        axRight, 'right-weights'), interval=AnimInterv)

    a11 = animation.FuncAnimation(fig2, drawWeights, fargs=(
        axLeftH, 'left-Head-weights'), interval=AnimInterv)
    a21 = animation.FuncAnimation(fig2, drawWeights, fargs=(
        axRightH, 'right-Head-weights'), interval=AnimInterv)

    # a3 = animation.FuncAnimation(
    #     fig, plot_dope, fargs=(axDope,), interval=AnimInterv)
    # a4 = animation.FuncAnimation(
    #     fig, plot_dope, fargs=(axDope2,), interval=AnimInterv)
    a5 = animation.FuncAnimation(
        fig, plot_angles, fargs=(axAngle,), interval=AnimInterv)
    a6 = animation.FuncAnimation(
        fig, plot_performance, fargs=(axPerf,), interval=AnimInterv)

    a7 = animation.FuncAnimation(
        fig4, drawWeights, fargs=(
            axHidden, 'hidden-weights'), interval=AnimInterv)
else:
    # axLeft = fig.add_subplot(421)
    # axRight = fig.add_subplot(422)
    # axDope = fig.add_subplot(423)
    # axDope2 = fig.add_subplot(424)
    # axAngle = fig.add_subplot(413)
    # axPerf = fig.add_subplot(414)

    drawWeights(0, axLeft, 'left-weights')
    drawWeights(0, axRight, 'right-weights')

    # drawWeights(0, axLeftH, 'left-Head-weights')
    # drawWeights(0, axRightH, 'right-Head-weights')

    # drawWeights(0, axHidden, 'hidden-weights')
    # drawWeights(0, axRightH, 'right-Head-weights')
    # plot_dope(0, axDope,)
    # plot_dope(0, axDope2,)
    plot_angles(0, axAngle,)
    plot_performance(0, axPerf,)

plt.show()
