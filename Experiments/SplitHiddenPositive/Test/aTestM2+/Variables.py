from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from gazebo_msgs.srv import SetModelState, GetModelState
import rospy
import numpy as np

ENBLE_LOGGING = False
# ENBLE_LOGGING = True

#----------------- DVS-Settings ----------------


dvsCutUP = 20
dvsCutDOWN = 50
dvsCutEND = 128 - dvsCutDOWN

dvsNumberOfX = 128
dvsNumberOfY = 128 - dvsCutDOWN - dvsCutUP  # 96 #128

BrainNumOfX = 5
BrainNumOfY = 2
numMonitorJoints = 3

globalRatio = ((float)(dvsNumberOfX*dvsNumberOfY)) / \
    (BrainNumOfX*BrainNumOfY)

xRatio = ((float)(dvsNumberOfX))/BrainNumOfX
yRatio = ((float)(dvsNumberOfY))/BrainNumOfY
numBrainNeurons = BrainNumOfX * BrainNumOfY
numJointNeurons = numMonitorJoints * 2
maxFrequ = 1 * globalRatio  # todo tweak settings
fullActivityThreshold = 1 * globalRatio  # 100

#-------------------Ball State Machine ----------

robotName = 'alphaS'
targetName = 'ball'

RobotStart = Point(-5., 0., 0.2)
maxDistance = 1.6
deltaDistance = 0.051

sphereRadius = 0.3
initialPos = Point(RobotStart.x+maxDistance, 0., sphereRadius)

m0 = Vector3(0.5, 0.0, 0.)  # 0.25, 0., 0.)

m1 = Vector3(2., 1.5, 0.)  # 1.5, 1.5, 0.)
m2 = Vector3(1., 0.0, 0.)  # 0.5, 0.0, 0.)
m3 = Vector3(1.5, -1.25, 0.)  # 1., -2.0, 0.)
m4 = Vector3(0., -0.5, 0.)  # -1.5, -2.0, 0.)
m5 = Vector3(-1.5, -1.25, 0.)  # -1.5, 0., 0.)
m6 = Vector3(-1., 0.0, 0.)  # -1.5, 2.0, 0.)
m7 = Vector3(-2., 1.5, 0.)  # -0.25, 0., 0.)

m8 = Vector3(-2., 1.5, 0.)  # -1.5, 1.5, 0.)
m9 = Vector3(-1., 0.0, 0.)  # -0.5, 0.0, 0.)
m10 = Vector3(-1.5, -1.25, 0.)  # -1., -2.0, 0.)
m12 = Vector3(0., -0.5, 0.)  # 1.5, -2.0, 0.)
m13 = Vector3(1.5, -1.25, 0.)  # 1.5, 0., 0.)
m14 = Vector3(1., 0.0, 0.)  # 1.5, 2.0, 0.)
m15 = Vector3(2., 1.5, 0.)

m16=Vector3(.5, 0.0, 0.)

t0=Vector3(1.,0.,0.)
t1=Vector3(1.5, 1.5, 0.)
t2=Vector3(1., 0., 0.)
t3=Vector3(1.5, -1.5, 0.)
t4=Vector3(1.,0.,0.)
t5=Vector3(1.5, 1.5, 0.)
t6=Vector3(1., 0., 0.)
t7=Vector3(1.5, -1.5, 0.)


moveArray = [t0, t1, t2, t3, t4, t5, t6, t7]  # [m0, m1, m2, m3, m4, m5, m6,m7, m8, m9, m10, m12, m13, m14, m15, m16]

moveStraight = Vector3(0.50, 0., 0.)  # Vector3(1.0, 0., 0.)
moveLeft = Vector3(3.0, 3., 0.)
moveRight = Vector3(0.50, 0., 0.)  # Vector3(3.0, -2., 0.)
moveTightLeft = Vector3(3.0, -3., 0.)  # Vector3(2., 3., 0.)
moveTightRight = Vector3(1.0, 0., 0.)  # Vector3(2., -3., 0.)

stepSize = 0.00018
get_model_state = rospy.ServiceProxy(
    '/gazebo/get_model_state', GetModelState, persistent=True)


#-------------------Ball State Machine Fuctions ----------

class SmachStates(object):
    def __init__(self, time, action, **kargs):
        self.time = time
        self.action = action
        self.args = kargs.items()

    def __str__(self):
        return "Time: {}, Action: {}, Additional: {}".format(str(self.time.to_sec()), self.action, self.args)

    def getString(self):
        return "Time: {}, Action: {}, Additional: {}".format(self.time.to_sec(), self.action, self.args)

    def getLog(self):
        return str(self.time.to_sec())+", "+self.action+", "+(("".join("{}:({}), ".format(str(key), ((str(value)).strip()).replace('\n', ' ')) for key, value in self.args))[:-2])
    RESET = "RESET"
    TARGET_DELETED = "TARGET_DELETED"
    TARGET_SPAWNED = "TARGET_SPAWNED"
    TARGET_WAITING = "TARGET_WAITING"
    TARGET_START_MOVING = "TARGET_START_MOVING"
    TARGET_MOVING = "TARGET_MOVING"
    TARGET_RESUMING = "TARGET_RESUMING"
    TARGET_AT_DESTINATION = "TARGET_AT_DESTINATION"


def Reset_smachState(time, roundNumber, robPos, ballPos, destination, inverted):
    return SmachStates(time, SmachStates.RESET, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos, destination=destination, inverted=inverted)


def DelTarget_smachState(time, roundNumber, robPos, ballPos):
    return SmachStates(time, SmachStates.TARGET_DELETED, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos)


def SpawnTarget_smachState(time, roundNumber, robPos, ballPos):
    return SmachStates(time, SmachStates.TARGET_SPAWNED, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos)


def TargetWaiting_smachState(time, roundNumber, robPos, ballPos, stepcounter):
    return SmachStates(time, SmachStates.TARGET_WAITING, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos, stepcounter=stepcounter)


def TargetMoving_smachState(time, roundNumber, robPos, ballPos, stepcounter):
    return SmachStates(time, SmachStates.TARGET_MOVING, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos, stepcounter=stepcounter)


def TargetStartMoving_smachState(time, roundNumber, robPos, ballPos, destination, inverted):
    return SmachStates(time, SmachStates.TARGET_START_MOVING, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos, destination=destination, inverted=inverted)


def TargetResuming_smachState(time, roundNumber, robPos, ballPos, stepcounter):
    return SmachStates(time, SmachStates.TARGET_RESUMING, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos, stepcounter=stepcounter)


def TargetAtDestination_smachState(time, roundNumber, robPos, ballPos):
    return SmachStates(time, SmachStates.TARGET_AT_DESTINATION, roundNumber=roundNumber, robPos=robPos, ballPos=ballPos)

#--------------------------------


def getTimeAndPosition():
    time = rospy.get_rostime()
    target = get_model_state(model_name=targetName)
    robot = get_model_state(model_name=robotName)
    return time, robot.pose.position, target.pose.position

#-------------------- dvs2Brain --------------


dvsMap = np.full((128, 128), -1.)
for x in range(dvsNumberOfX):
    for y in range(dvsCutUP, dvsCutUP+dvsNumberOfY):
        yNew = y - dvsCutUP
        outPutX = np.floor(x/xRatio)
        outPutY = np.floor(yNew / yRatio)
        inversOutPutY = BrainNumOfY - outPutY
        dvsMap[y][x] = (inversOutPutY - 1) * BrainNumOfX + outPutX


idMap = [Point(x=(i % BrainNumOfX)+1, y=((i - ((i % BrainNumOfX)+1))/BrainNumOfX)+2)
         for i in range(0, numBrainNeurons)]


def eventMapper(event):
    return dvsMap[event.y][event.x]


def idMapper(ID):
    return idMap[ID-1]


# dvsMap = np.zeros((108, 128))
# for x in range(128):
#     for y in range(20, 108):
#         yNew = y - 20
#         outPutX = np.ceil(x/8)
#         outPutY = np.floor(yNew / 8)
#         inversOutPutY = 11 - outPutY
#         dvsMap[y][x] = (inversOutPutY - 1) * 16 + outPutX
