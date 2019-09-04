# Imported Python Transfer Function
"""
This module contains the transfer function that transforms DVS address events
to spikes and input them to a population of neurons
"""
from dvs_msgs.msg import EventArray
import numpy as np
@nrp.MapRobotPublisher('dvs_rendered', Topic('/dvs_rendered_detection', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber('dvs', Topic('head/dvs128/events', EventArray))
@nrp.Robot2Neuron()
def renderDvs(t, dvs, dvs_rendered):
    import Variables as Var
    event_msg = dvs.value
    if event_msg is None:
        return
    rendered_img = np.zeros((128, 128, 3), dtype=np.uint8)
    ySteps = int(np.floor(Var.yRatio))
    xSteps = int(np.floor(Var.xRatio))
    for y in range(Var.dvsCutUP,Var.dvsCutEND+1,ySteps):
        # for x in range(0,128):  
            # lines from left to right
            rendered_img[y] = [(0,0,255) for i in range(128)]
    for x in range(0, 128, xSteps):
        for y in range(Var.dvsCutUP, Var.dvsCutEND):
            rendered_img[y][x]= (0,0,255)
            #rendered_img[j][i]= (0,0,255)
    # for x in range(0,128):
    #     # for y in range(128):
    #         rendered_img[x][x]=(0,255,0)

    for event in event_msg.events:
        # temp = rendered_img[event.y][event.x][0]
        # rendered_img[event.y][event.x] = (max(255,temp+5), 255, 0)
        rendered_img[event.y][event.x] = (event.polarity * 255, 255, 0)
    msg_frame = CvBridge().cv2_to_imgmsg(rendered_img, 'rgb8')
    dvs_rendered.send_message(msg_frame)
