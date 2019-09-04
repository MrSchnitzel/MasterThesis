import hbp_nrp_cle.tf_framework as nrp
import numpy as np

header = ['DATA']# ['time','action','invert','direction']
# @MapCSVRecorder  
# todo check  for example $HBP/Experiments/braitenberg_husky/csv_robot_position.py.
@nrp.MapCSVRecorder("stateRecorder", filename="state.csv", headers=header)
@nrp.MapRobotSubscriber("smachLogger", Topic("/log/smach_log", std_msgs.msg.String))
@nrp.Neuron2Robot(triggers='smachLogger')

def stateLogger(t,stateRecorder,smachLogger):
    if smachLogger.value is None:
        return
    if smachLogger.changed is False:
        return
    msg = smachLogger.value.data
    msg = [(s.strip()).replace('\n',' ') for s in msg.split(',')]
    # clientLogger.info(dir(smachLogger))
    apply(stateRecorder.record_entry,msg)
    smachLogger.reset_changed()