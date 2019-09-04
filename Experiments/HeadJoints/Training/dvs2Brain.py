from dvs_msgs.msg import EventArray
import numpy as np


@nrp.MapRobotSubscriber('dvs', Topic('head/dvs128/events', EventArray))
@nrp.MapRobotSubscriber("lastTerm_sub", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapSpikeSource('sensor_neurons', nrp.map_neurons(range(0, 10), lambda i: nrp.brain.sensors[i]), nrp.poisson)
@nrp.Robot2Neuron()
def dvs2Brain(t, lastTerm_sub, dvs, sensor_neurons):
    import Variables as VAR
    event_msg = dvs.value
    if event_msg is None:
        return
    if lastTerm_sub.value is None or t < lastTerm_sub.value.data + 2.:
        for i in range(VAR.numBrainNeurons):
            sensor_neurons[i].rate = 0.
        return

    dvsEvents = event_msg.events
    brainPotentials = np.zeros(VAR.numBrainNeurons, dtype=np.uint8)
    # dvsEvents = list(filter(lambda event: event.y < VAR.dvsNumberOfY,dvsEvents))
    dvsMap = VAR.dvsMap
    for event in dvsEvents:
        #brainNeuron = mapDvsP2BrainP(event.x,event.y)
        brainNeuron = dvsMap[event.y][event.x]
        if brainNeuron >= 0:
            brainPotentials[brainNeuron] = brainPotentials[brainNeuron] + 1

    
    rates=[]
    for neuronID in range(VAR.numBrainNeurons):
        neuronEvents = brainPotentials[neuronID]

        # neuronFrequ = (float(neuronEvents) /
        #                VAR.globalRatio)*40
        # neuronFrequ = min(neuronFrequ, 40)
        rates.append(neuronEvents * 50.)
        sensor_neurons[neuronID].rate= neuronEvents * 50. #min(100,neuronEvents * 5.) # neuronFrequ
        # if t % 1 > 0.02:
        #     clientLogger.info(neuronID, " freq: ", neuronFrequ)
    # if t % 2 < 0.02:
    #     clientLogger.info(rates)
    return sensor_neurons
