from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
import std_msgs
# from tf.transformations import *

import numpy as np


@nrp.MapCSVRecorder("PerformanceRecorder", filename="performance.csv", headers=['time', "direction", "directionHead", "Angle of Head to Target", "Angle of Body to Target", "Raw Angle to Target", "Raw Angle of Body to Target", "Distance to Target"])
@nrp.MapRobotSubscriber("radHead", Topic("/alphaS/joint_1/cmd_pos", std_msgs.msg.Float64))
# .Int32MultiArray))
@nrp.MapRobotSubscriber("model_states", Topic("gazebo/model_states", ModelStates))
@nrp.MapRobotPublisher("terminator", Topic("robot/terminate_flag", std_msgs.msg.Bool))
@nrp.MapRobotSubscriber("direct", Topic("/brain/direction", std_msgs.msg.Float32))
@nrp.MapRobotSubscriber("directHead", Topic("/alphaS/offsetHead", std_msgs.msg.Float64))
# .Int32MultiArray))
@nrp.MapRobotPublisher("lastTerm_writer", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("lastTerm_sub", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("angle_writer", Topic("robot/angle", std_msgs.msg.Float32MultiArray))
@nrp.MapRobotSubscriber("angle_sub", Topic("robot/angle", std_msgs.msg.Float32MultiArray))
@nrp.MapRobotPublisher("body_angle_writer", Topic("robot/body_angle", std_msgs.msg.Float32MultiArray))
@nrp.MapRobotSubscriber("body_angle_sub", Topic("robot/body_angle", std_msgs.msg.Float32MultiArray))
# @nrp.MapSpikeSource("dopamin_neurons", nrp.map_neurons(range(0, 2), lambda i: nrp.brain.dopamin[i]), nrp.poisson)
# @nrp.MapSpikeSource("dopDown_neurons", nrp.map_neurons(range(0, 2), lambda i: nrp.brain.dopaminDown[i]), nrp.poisson)
@nrp.MapRobotPublisher("dOut", Topic("/brain/direction", std_msgs.msg.Float32))
@nrp.MapRobotPublisher("offset", Topic("/alphaS/offset", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("offsetHead", Topic("/alphaS/offsetHead", std_msgs.msg.Float64))
# @nrp.MapSpikeSource("ballPos", nrp.map_neurons(range(0, 2), lambda i: nrp.brain.output_layer[i]), nrp.poisson, weight=100.0)
@nrp.Robot2Neuron()
def rewardTF(t, direct, directHead, dOut, offset, offsetHead, angle_writer, angle_sub, body_angle_writer, body_angle_sub, radHead, lastTerm_writer, lastTerm_sub, model_states, PerformanceRecorder, terminator):
    arraySize = 625  # 4*500
    arrayBody = 625  # 4*500

    deltaAngleReward = 0.
    angleAbort = 35.

    deltaAngleRewardH = 0.
    angleAbortH = 40.
    import nest

    import Variables as VAR
    from vc import vec
    import tf
    import math
    if lastTerm_sub.value is None:
        lastTerm_writer.send_message(std_msgs.msg.Float64(0.0))

    if angle_sub.value is None or body_angle_sub.value is None:
        angle_writer.send_message(std_msgs.msg.Float32MultiArray(
            data=[0. for i in range(arraySize)]))
        body_angle_writer.send_message(std_msgs.msg.Float32MultiArray(
            data=[0. for i in range(arrayBody)]))
        # angle_writer.send_message(std_msgs.msg.Float32MultiArray([0.0 in range(arraySize)]))
        return

    if model_states.value is None or len(model_states.value.pose) < 2 or radHead.value is None:
        return

    radHead = radHead.value.data
    model_states = model_states.value

    robot = model_states.pose[0]
    rp = robot.position
    ro = robot.orientation

    ball = model_states.pose[1]
    bp = ball.position

    qRobot = list(tf.transformations.quaternion_inverse(
        [ro.x, ro.y, ro.z, ro.w]))
    qRobotConj = tf.transformations.quaternion_conjugate(qRobot)

    T0R = [[1, 0, 0, -rp.x], [0, 1, 0, -rp.y],
           [0, 0, 1, -rp.z], [0, 0, 0, 1]]

    nppBallTrans = list(np.dot(T0R, [bp.x, bp.y, bp.z, 1]))
    npBall = tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(qRobot, nppBallTrans), qRobotConj)

    xAxies = vec(1, 0)
    npBall2d = vec(npBall[0], npBall[1])

    angle = xAxies.findClockwiseAngle180(npBall2d)

    angleRaw = angle

    oH = 0.
    if directHead.value is not None:
        oH = directHead.value.data

    bodyAngleRaw = angleRaw + oH*90.  # + (radHead * 180 / np.pi)

    temp = list(angle_sub.value.data)
    temp[int(math.floor((t * 50) % arraySize))] = angleRaw
    tempBody = list(body_angle_sub.value.data)
    # meanBodyAngleBefore = reduce(lambda a, b: a + b, tempBody) / len(tempBody)
    tempBody[int(math.floor((t * 50) % arrayBody))] = bodyAngleRaw

    angle_writer.send_message(std_msgs.msg.Float32MultiArray(data=temp))
    body_angle_writer.send_message(
        std_msgs.msg.Float32MultiArray(data=tempBody))

    meanAngle = reduce(lambda a, b: a + b, temp) / len(temp)
    meanBodyAngle = reduce(lambda a, b: a + b, tempBody) / len(tempBody)

    # angleTrust = 0.0
    # bodyAngleTrust = 0.0
    # angle = angleTrust * angleRaw + (1 - angleTrust) * meanAngle
    # bodyAngle = bodyAngleTrust * bodyAngleRaw + (1 - bodyAngleTrust) * meanBodyAngle

    l = nrp.config.brain_root.conn_l
    r = nrp.config.brain_root.conn_r

    lH = nrp.config.brain_root.conn_lH
    rH = nrp.config.brain_root.conn_rH

    if t % 1 < 0.02:
        distanceRB = np.sqrt(nppBallTrans[0] ** 2 + nppBallTrans[1] ** 2)
        direction = 0.
        directionHead = 0.
        if direct.value:
            direction = direct.value.data
        if directHead.value:
            directHead = directHead.value.data
        PerformanceRecorder.record_entry(
            t, direction, directHead, meanAngle, meanBodyAngle, angleRaw, bodyAngleRaw, distanceRB)

    # if VAR.ENBLE_LOGGING and t % 3 < 0.02:
    #     clientLogger.info("----------------Reward Function----------------")
    #     clientLogger.info('Current Angle:', angle)
    #     clientLogger.info('Current Distance: ', distanceRB)
        # angle_writer.send_message(std_msgs.msg.Float64(0.))
        # if(np.absolute(angle) > 50):
        #     clientLogger.info("angle to big! (>50)")
        # clientLogger.info(angleBody)

    if t < lastTerm_sub.value.data + 0.2:
        angle_writer.send_message(std_msgs.msg.Float32MultiArray(
            data=[0. for i in range(arraySize)]))
        body_angle_writer.send_message(std_msgs.msg.Float32MultiArray(
            data=[0. for i in range(arrayBody)]))
        angle = 0.
        bodyAngleRaw = 0.
        # bodyAngle = 0.

    # if t % 200 < 0.02:
    #     lastTerm_writer.send_message(std_msgs.msg.Float64(t-3))

    # ----------------------------------
    # ---------- Reward Setting for Body
    judgedAngle = bodyAngleRaw

    # if t < lastTerm_sub.value.data + 10.5:
    #     judgedAngle = meanAngle

    if(np.absolute(meanBodyAngle) > angleAbort):
        if(t > lastTerm_sub.value.data+6.):
            angle_writer.send_message(std_msgs.msg.Float32MultiArray(
                data=[0. for i in range(arraySize)]))
            body_angle_writer.send_message(std_msgs.msg.Float32MultiArray(
                data=[0. for i in range(arrayBody)]))
            dOut.send_message(0.)

            offset.send_message(0.)
            offsetHead.send_message(0.)

            lastTerm_writer.send_message(std_msgs.msg.Float64(t))
            terminator.send_message(std_msgs.msg.Bool(True))
            return

    hiddenLayer = nrp.config.brain_root.hidden_layer
    hiddenLayer_left = nrp.config.brain_root.hidden_layer_left
    hiddenLayer_right = nrp.config.brain_root.hidden_layer_right
    # or (t < lastTerm_sub.value.data + 12.5):
    if(np.absolute(judgedAngle) < deltaAngleReward):
        nest.SetStatus(l, {"n": 0.})
        nest.SetStatus(r, {"n": 0.})
        # nest.SetStatus(l, {"c": 0.})
        # nest.SetStatus(r, {"c":  0.})

        for j in hiddenLayer:
            # outConnections = nest.GetConnections(source=[j])
            # weights = nest.GetStatus(outConnections, keys="weight")
            reward = 0.
            inCon = nest.GetConnections(target=[j])
            # nest.SetStatus(inCon, {"c": 0.})
            nest.SetStatus(inCon, {"n": reward})
    else:
        deltaTimeRewardReduce = 3000
        rewardFac = 0.0001  # 0.00000515
        if t > 1 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 2 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 3 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 4 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 5 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5

        # reward = meanBodyAngleBefore - meanBodyAngle

        if judgedAngle > 0:
            judgedAngle = judgedAngle - deltaAngleReward
        else:
            judgedAngle = judgedAngle + deltaAngleReward

        # normAngle = judgedAngle / (angleAbort - deltaAngleReward)
        globReward = judgedAngle * rewardFac
        # globReward = ((2/(1+math.e**(-normAngle*8)))-1) * \
        #     rewardFac  # 5 maybe insead of 8

        # (-meanBodyAngle * 0.5 * rewardFac)})
        nest.SetStatus(l, {"n": -globReward})
        # ( meanBodyAngle * 0.5 * rewardFac)})
        nest.SetStatus(r, {"n":  globReward})
        # nest.SetStatus(l, {"c": -globReward })# (-meanBodyAngle * 0.5 * rewardFac)})
        # nest.SetStatus(r, {"c":  globReward })# ( meanBodyAngle * 0.5 * rewardFac)})

        # nValuesHidden = []
        # rweights = nest.GetStatus(r, keys="weight")
        # # maxR = reduce(lambda x, y: max(abs(x), abs(y)), rweights)/len(rweights)
        # maxR = reduce(lambda x, y: max(abs(x), abs(y)), rweights)
        for j in hiddenLayer_right:
            outConnections = nest.GetConnections(source=[j])
            weights = nest.GetStatus(outConnections, keys="weight")

            reward = globReward * weights[1]/abs(weights[1])#/maxR
            inCon = nest.GetConnections(target=[j])

            nest.SetStatus(inCon, {"n": reward})

        # lweights = nest.GetStatus(l, keys="weight")
        # # maxL = reduce(lambda x, y: abs(x) + abs(y), lweights)/len(lweights)
        # maxL = reduce(lambda x, y: max(abs(x), abs(y)), lweights)
        for j in hiddenLayer_left:
            outConnections = nest.GetConnections(source=[j])
            weights = nest.GetStatus(outConnections, keys="weight")

            # #todo: check if "max(weights)" must be set to use abs value when using with negative numbers
            # reward = (
            #     globReward * (weights[1]-weights[2]))/(max(map(lambda w: abs(w), weights[-2:])) * 2)

            reward = -globReward * weights[1]/abs(weights[1])
            # if t % 2 < 0.02:
            #     clientLogger.info(lweights)
            #     clientLogger.info(weights)

            inCon = nest.GetConnections(target=[j])

            nest.SetStatus(inCon, {"n": reward})            
                    # nValuesHidden.append(reward)
                    # nest.SetStatus(inCon, {"c": 0.})

    # ----------------------------------
    # ---------- Reward Setting 50r Head
    judgedAngleH = angleRaw
    if(np.absolute(judgedAngleH) > angleAbortH):
        if(t > lastTerm_sub.value.data+2.5):
            angle_writer.send_message(std_msgs.msg.Float32MultiArray(
                data=[0. for i in range(arraySize)]))
            body_angle_writer.send_message(std_msgs.msg.Float32MultiArray(
                data=[0. for i in range(arrayBody)]))
            dOut.send_message(0.)

            offset.send_message(0.)
            offsetHead.send_message(0.)

            lastTerm_writer.send_message(std_msgs.msg.Float64(t))
            terminator.send_message(std_msgs.msg.Bool(True))
            return
    if(np.absolute(judgedAngleH) < deltaAngleRewardH or t < lastTerm_sub.value.data + 0.4):
        nest.SetStatus(lH, {"n": 0.})
        nest.SetStatus(rH, {"n": 0.})
    else:
        deltaTimeRewardReduce = 5700
        rewardFac = 0.0000415
        if t > 1 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 2 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5
        if t > 3 * deltaTimeRewardReduce:
            rewardFac = rewardFac * 0.5

        if judgedAngleH > 0:
            judgedAngleH = judgedAngleH - deltaAngleRewardH
        else:
            judgedAngleH = judgedAngleH + deltaAngleRewardH

        nest.SetStatus(lH, {"n": -judgedAngleH * rewardFac})
        nest.SetStatus(rH, {"n":  judgedAngleH * rewardFac})
