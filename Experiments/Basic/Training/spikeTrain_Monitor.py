@nrp.NeuronMonitor(nrp.brain.input_Head[slice(0,10 , 1)], nrp.spike_recorder)
def spikeTrain_Monitor (t):
    #import tf
    #t = tf.TransformerROS()
    #clientLogger.info(t.getFrameStrings())
    return True    
