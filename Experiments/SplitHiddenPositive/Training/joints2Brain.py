from dvs_msgs.msg import EventArray
import numpy as np


@nrp.MapRobotSubscriber("radHead", Topic("/alphaS/joint_1/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad1", Topic("/alphaS/joint_1/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad3", Topic("/alphaS/joint_3/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad5", Topic("/alphaS/joint_5/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad7", Topic("/alphaS/joint_7/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad9", Topic("/alphaS/joint_9/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad11", Topic("/alphaS/joint_11/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad13", Topic("/alphaS/joint_13/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("rad15", Topic("/alphaS/joint_15/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("lastTerm_sub", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapSpikeSource('joint_neurons', nrp.map_neurons(range(0, 3*2), lambda i: nrp.brain.joints[i]), nrp.poisson)
@nrp.Robot2Neuron()
def joints2Brain(t, lastTerm_sub, radHead, rad1, rad3, rad5, rad7, rad9, rad11, rad13, rad15, joint_neurons):
    import Variables as Var
    if lastTerm_sub.value is None or t < lastTerm_sub.value.data + 2 or radHead.value is None:
        for i in range(Var.numJointNeurons):
            joint_neurons[i].rate = 0.
        return
    jointAngleAr = [radHead, rad1, rad3, rad5, rad7, rad9, rad11, rad13, rad15]

    for i in range(len(jointAngleAr)):
        jointAngleAr[i] = jointAngleAr[i].value.data * 180/np.pi
    # clientLogger.info(jointAngleAr)

    # brainPotentials = [[0.,0.] for i in range(Var.numMonitorJoints)]

    #clientLogger.info(brainPotentials)
    result = []
    for jointId in range(Var.numMonitorJoints):

        jA = jointAngleAr[jointId]
        if jA > 90:
            jA = 90
        if jA < -90:
            jA = -90

        #map to [-1,1]
        normAngle = (jA / 90.)
        #map to [0,2]
        # normAngle = 1 + normAngle

        # r = max(0., 0.5*normAngle)
        # l=max(0.,- 0.5 * normAngle)
        # if Var.ENBLE_LOGGING and t%2<0.03:
        #     clientLogger.info(jA,'->', normAngle,'->',r,'/',l)

        neuronFrequR = max(0., normAngle*5000)
        neuronFrequL = max(0., - normAngle*5000)

        # neuronFrequR = min(r*10, 10)
        # neuronFrequL = min(l*10, 10)
        result.append((neuronFrequR, neuronFrequL))
        joint_neurons[jointId*2].rate = neuronFrequR
        joint_neurons[(jointId * 2) + 1].rate = neuronFrequL

    # if t % 2 < 0.02:
    #     clientLogger.info(result)
    return joint_neurons
