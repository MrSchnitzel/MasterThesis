@nrp.NeuronMonitor(nrp.brain.circuit[slice(0,32 , 1)], nrp.spike_recorder)
def spikeTrain_Monitor (t):
    #import tf
    #t = tf.TransformerROS()
    #clientLogger.info(t.getFrameStrings())
    return True    
