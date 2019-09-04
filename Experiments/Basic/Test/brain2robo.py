import numpy as np
@nrp.MapRobotPublisher("offset", Topic("/alphaS/offset", std_msgs.msg.Float64))
# @nrp.MapRobotPublisher("offsetHead", Topic("/alphaS/offsetHead", std_msgs.msg.Float64))
# @nrp.MapSpikeSink("ballPos", nrp.map_neurons(range(0,2), lambda i: nrp.brain.output_layer[i]), nrp.leaky_integrator_alpha)

# @nrp.MapSpikeSink("right", nrp.brain.output_layer[0], nrp.spike_recorder)#, use_ids = True)
# @nrp.MapSpikeSink("left", nrp.brain.output_layer[1], nrp.spike_recorder)

@nrp.MapRobotPublisher("dOut", Topic("/brain/direction", std_msgs.msg.Float32))
@nrp.MapRobotSubscriber("direction", Topic("/brain/direction", std_msgs.msg.Float32))
# @nrp.MapRobotSubscriber("directionHead", Topic("/alphaS/offsetHead", std_msgs.msg.Float64))

@nrp.MapRobotSubscriber("lastTerm_sub", Topic("robot/lastTermTime", std_msgs.msg.Float64))

#rechtes neuron sagt Ball rechts
# ,left,right):#, leftSCount, rightSCOut,leftSCOut):
def brain2robo(t, offset, direction, dOut, lastTerm_sub):
        import math
        import Variables as VAR
        import nest
        lastTerm = 0.
        if lastTerm_sub.value is not None:
          lastTerm = lastTerm_sub.value.data
        if direction.value is None or lastTerm + 0.03 > t:
          out = 0.
        else:
          out = direction.value.data
        
        spike_detector = nrp.config.brain_root.spike_detector

        test = nest.GetStatus(spike_detector, keys="n_events")
       	nest.SetStatus(spike_detector, {"n_events": 0})
        
        # summ = test[0] + test[1] +1
        r = test[0]
        l = test[1]
        
        delt = r - l
        meanCount = float(l + r) #/ 2.0
        if meanCount > 0:
          direc = delt/meanCount
        else:
          direc = 0.

        out = out + direc*0.05  # meanCount * delt * 0.5 + (1-meanCount)*out


        out = np.clip(out, -1., 1.)
        if not VAR.ENBLE_LOGGING and t % 2 < 0.02:
          clientLogger.info("-----------------Robo Controll-----------------")
          clientLogger.info('Direction: ', out*0.3)
          clientLogger.info('DirectionHead: ', 0.)
          clientLogger.info('count: ', test)
          clientLogger.info('direc: ', direc)
          clientLogger.info('direcHead: ', 0.)
          # if t % 6 < 0.02 :
          #   clientLogger.info("MotorNeuron Voltage:",ballPos.voltage)
          # clientLogger.info(str(ballPos[0].voltage)+" /// "+str(ballPos[1].voltage))
        dOut.send_message(out)
        offset.send_message(out*0.3)
        
          # offset.send_message(np.power(out,3)*0.38)
