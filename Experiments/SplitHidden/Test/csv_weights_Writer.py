import hbp_nrp_cle.tf_framework as nrp
import numpy as np

nrp.config.brain_root.conn_l
nrp.config.brain_root.conn_r
header = ['time'] + map(str, range(40))
# @MapCSVRecorder
# todo check  for example $HBP/Experiments/braitenberg_husky/csv_robot_position.py.
@nrp.MapCSVRecorder("DopeRecorder", filename="dope.csv", headers=['time', "concentrationR", "concentrationL", 'headR', 'headL'])
# @nrp.MapCSVRecorder("leftDopeRecorder", filename="left-dope.csv", headers=header)
@nrp.MapCSVRecorder("rightRecorder", filename="right-weights.csv", headers=header)
@nrp.MapCSVRecorder("leftRecorder", filename="left-weights.csv", headers=header)
@nrp.MapCSVRecorder("rightHeadRecorder", filename="right-Head-weights.csv", headers=header)
@nrp.MapCSVRecorder("leftHeadRecorder", filename="left-Head-weights.csv", headers=header)
@nrp.MapCSVRecorder("hiddenRec", filename="hidden-weights.csv", headers=header)
@nrp.Neuron2Robot(throttling_rate=1.)
def csv_weights_Writer(t, rightRecorder, leftRecorder, rightHeadRecorder, leftHeadRecorder, hiddenRec, DopeRecorder):
    import nest

    nL = nest.GetStatus(nrp.config.brain_root.conn_l, keys="n")
    nR = nest.GetStatus(nrp.config.brain_root.conn_r, keys="n")

    nLH = nest.GetStatus(nrp.config.brain_root.conn_lH, keys="n")
    nRH = nest.GetStatus(nrp.config.brain_root.conn_rH, keys="n")

    # if t % 0.5 < 0.02:
    #     clientLogger.info("----------------")
    #     clientLogger.info(nR[0])
    #     clientLogger.info(nL[0])

#     clientLogger.info("left:", a)

    a = nest.GetStatus(nrp.config.brain_root.conn_l, keys="weight")
    b = nest.GetStatus(nrp.config.brain_root.conn_r, keys="weight")
    hiddenConnections = nest.GetConnections(
        target=map(int, nrp.config.brain_root.hidden_layer.all_cells))
    weightsHidden = nest.GetStatus(hiddenConnections, keys="weight")
    c = nest.GetStatus(nrp.config.brain_root.conn_lH, keys="weight")
    d = nest.GetStatus(nrp.config.brain_root.conn_rH, keys="weight")
    # clientLogger.info("right:", str(b))

    # if t % 1 < 0.02:
    largs = ((t,) + a)
    rargs = ((t,) + b)
    hiddenArgs = ((t,)+weightsHidden)
    lHargs = ((t,) + c)
    rHargs = ((t,) + d)
    DopeRecorder.record_entry(t, nR[0], nL[0], nRH[0], nLH[0])
    apply(leftRecorder.record_entry, largs)
    apply(rightRecorder.record_entry, rargs)
    apply(hiddenRec.record_entry, hiddenArgs)
    apply(leftHeadRecorder.record_entry, lHargs)
    apply(rightHeadRecorder.record_entry, rHargs)
