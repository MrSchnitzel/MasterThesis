# from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging
import nest
# from .resources import Variables as Var
import pyNN.nest as sim
from pyNN.nest import *
from hbp_nrp_excontrol.logs import clientLogger
import h5py

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

nest.ResetKernel()
sim.setup(threads=1)

"""
Initializes PyNN with the neuronal network that has to be simulated
"""

# R-STDP parameters
# Minimum weight value
w_max = 6000.
# Maximum weight value
w_min = -6000.
# Maximum initial random value
w0_max = 700.
# Minimum initial random value
w0_min = -200.
# Time constant of reward signal in ms
tau_n_out = 5.  # 200.
tau_n_hidden = 5.  # 200.
# Time constant of eligibility trace in ms
tau_c_out = 10.  # 100.  # 1000.
tau_c_hidden = 10.  # 100.  # 1000.
# Factor that dopamine modulator for turning is multiplied with
# max_turning_dopamine_factor = 0.0015
# Factor that dopamine modulator for speed is multiplied with
# max_speed_dopamine_factor = 0.001

# Constant scaling strength of potentiation
A_plus_out = 1.
A_plus_hidden = 1.
# Constant scaling strength of depression
A_minus_out = -1.
A_minus_hidden = -1.

############################
###### CREATE LAYERS #######
############################


# INPUT_PARAMS = {
#     'cm': 1.0,
#     'e_rev_E': 0.0,
#     'e_rev_I': -70.0,
#     'i_offset': 0.0,
#     'tau_m': 20.0,
#     'tau_refrac': 0.1,
#     'tau_syn_E': 0.3,
#     'tau_syn_I': 0.5,
#     'v_reset': -65.0,
#     'v_rest': -65.0,
#     'v_thresh': -50.0}

INPUT_PARAMS = {
    'v_rest': -65.0,
    'cm': 1.0,
    'tau_m': 20.0,
    'tau_refrac': 0.,
    'tau_syn_E':  2.5,  # 10.,  # 2.5,
    'tau_syn_I':  2.5,  # 10.,  # 2.5,
    # 'e_rev_E':  0.0,
    # 'e_rev_I': -70.0,
    'v_thresh': -64.50,
    'v_reset': -65.0,
    'i_offset': 0.0,
}

#     'cm': 0.025,
#     'e_rev_E': 0.0,
#     'e_rev_I': -75.0,
#     'tau_m': 10.,
#     # brain sim time length 10 ms -> max rate 100
#     'tau_refrac': 0.10,
#     'tau_syn_E': 2.5,
#     'tau_syn_I': 2.5,
#     'v_reset': -60.5,
#     'v_rest': -60.5,
#     'v_thresh': -60.0,

HIDDEN_PARAMS = {
    'v_rest': -65.0,
    'cm': 100.0,
    'tau_m': 50.0,
    'tau_refrac': 1.,
    'tau_syn_E':  2.5,  # 10.,  # 2.5,
    'tau_syn_I':  2.5,  # 10.,  # 2.5,
    # 'e_rev_E':  0.0,
    # 'e_rev_I': -65.0,
    'v_thresh': -50.0,
    'v_reset': -65.0,
    'i_offset': 0.0,
}

OUTPUT_PARAMS = {
    'v_rest': -65.0,
    'cm': 10.0,
    'tau_m': 50.0,
    'tau_refrac': 1.,
    'tau_syn_E':  2.5,  # 10.,  # 2.5,
    'tau_syn_I':  2.5,  # 10.,  # 2.5,
    # 'e_rev_E':  0.0,
    # 'e_rev_I': -65.0,
    'v_thresh': -50.0,
    'v_reset': -65.0,
    'i_offset': 0.0,
}


output_layer_Body = sim.Population(2, sim.IF_cond_alpha, OUTPUT_PARAMS)
output_layer_Head = sim.Population(2, sim.IF_cond_alpha,  OUTPUT_PARAMS)
input_dvs = sim.Population(10, sim.IF_cond_alpha, INPUT_PARAMS)
input_jointsHead = sim.Population(3*2, sim.IF_cond_alpha, INPUT_PARAMS)
hidden_layer = sim.Population(8, sim.IF_cond_alpha, HIDDEN_PARAMS)
# hidden_layer_right = sim.Population(6, sim.IF_cond_alpha, HIDDEN_PARAMS)
# hidden_layer_left = sim.Population(6, sim.IF_cond_alpha, HIDDEN_PARAMS)
# input_jointsRest = sim.Population(6*2, sim.IF_cond_alpha, INPUT_PARAMS)
# trans_layer = sim.Population(2, sim.IF_cond_alpha,TRANS_PARAMS)


# ovserve = output_layer + input_Body

vtLeft = nest.Create('volume_transmitter', 1)
vtRight = nest.Create('volume_transmitter', 1)

spike_detector = nest.Create("spike_detector", 4, params={
    "withgid": True, "withtime": True})


############################
###### CONNECT LAYERS ######
############################
nest.CopyModel('stdp_dopamine_synapse', 'syn_Ouput',
               {'Wmax': w_max,
                'Wmin': w_min,
                'A_plus': 0.2,
                'A_minus': -0.1,
                'tau_c': 100.,
                'tau_n': 20.,  # ,
                'b': 0.,
                'vt': vtRight[0]
                })
nest.CopyModel('stdp_dopamine_synapse', 'syn_Ouput_Head',
               {'Wmax': w_max,
                'Wmin': w_min,
                'A_plus': 0.2,
                'A_minus': -0.1,
                'tau_c': 40.,
                'tau_n': 20.,  # ,
                'b': 0.,
                   'vt': vtRight[0]
                })

nest.CopyModel('stdp_dopamine_synapse', 'syn_Hidden',
               {'Wmax': w_max,
                'Wmin': w_min,
                'A_plus': 0.002,  # A_plus,
                'A_minus': -0.001,  # A_minus,
                'tau_c': 10000.,
                'tau_n': 20.,  # ,
                'b': 0.,
                'vt': vtRight[0]
                })


input_joints = input_jointsHead  # + input_jointsRest
output_layer = output_layer_Body + output_layer_Head


input_Head = input_dvs  # + input_jointsHead
input_hidden = output_layer_Head + input_joints
# input_Body = hidden_layer

circuit = output_layer + hidden_layer + input_dvs + input_joints

syn_con_dvsHead = nest.Connect(map(int, input_Head.all_cells),
                               map(int, output_layer_Head.all_cells),
                               syn_spec={'model': 'syn_Ouput_Head', 'weight': {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})
# syn_spec = {'model': 'syn_Ouput', 'weight': {'distribution': 'normal_clipped', 'low': 1., 'high': 400., 'mu': 200.0, 'sigma': 200.0}, 'delay': 1.0})
#syn_spec = {'model': 'syn_Hidden', 'weight': 100., 'delay': 1.0})  # {'distribution': 'normal_clipped', 'low': 2., 'high': 400., 'mu': 50.0, 'sigma': 300.0}, 'delay': 1.0})
# temp = map(int, hidden_layer.all_cells)
# temp.reverse()
syn_con_inHidden = nest.Connect(map(int, input_hidden.all_cells),
                                map(int, hidden_layer.all_cells),
                                syn_spec={'model': 'syn_Hidden', 'weight':  {'distribution': 'uniform', 'low': 10., 'high': w0_max}, 'delay': 1.0})
# syn_con_inHidden = nest.Connect(map(int, input_hidden.all_cells),
#                                 map(int, hidden_layer_left.all_cells),
#                                 syn_spec={'model': 'syn_Hidden', 'weight':  {'distribution': 'uniform', 'low': 10., 'high': w0_max}, 'delay': 1.0})
#    {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}
syn_con_right = nest.Connect(map(int, hidden_layer.all_cells),
                             map(int, output_layer_Body.all_cells[0:1]),
                             syn_spec={'model': 'syn_Ouput', 'weight':   {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})  # {'distribution': 'normal', 'mu': 20300.0, 'sigma': 800.0}, 'delay': 1.0})
syn_con_left = nest.Connect(map(int, hidden_layer.all_cells),
                            map(int, output_layer_Body.all_cells[1:2]),
                            syn_spec={'model': 'syn_Ouput', 'weight':   {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})  # {'distribution': 'normal', 'mu': 300.0, 'sigma': 200.0}, 'delay': 1.0})
#  syn_spec={'model': 'syn_Ouput', 'weight': {'distribution': 'normal', 'mu': 0.0, 'sigma': 300.0}, 'delay': 1.0})

# hidden_layer = hidden_layer_right + hidden_layer_left

nest.Connect(map(int, output_layer),
             map(int, spike_detector), "one_to_one")

conn_r = nest.GetConnections(target=[output_layer[0]])
conn_l = nest.GetConnections(target=[output_layer[1]])

conn_rH = nest.GetConnections(target=[output_layer[2]])
conn_lH = nest.GetConnections(target=[output_layer[3]])


conn_hidden = nest.GetConnections(
    target=[hidden_layer[i] for i in range(len(hidden_layer))])

#copy weights array from csv in variable
weightsConR = [715.667062537, 1195.70439155, 1251.24001291,
               55.4559020097, -1087.7835497, 770.12689139, 387.520509253, -182.241539376]
weightsConRH = [353.260597604, -592.730956036, 47.5028326638, 956.154026755,
                75.8888660702, -106.153802207, -484.526850428, 278.605692475, 788.637626401, 346.728054459]
weightsConL = [259.31292162, 167.4659756, -79.2767103931, 1451.69819076, 2052.4831168, 396.145181021, 574.24116668, 1212.45436104]
weightsConLH = [-16.6883127395, 475.321685825, 56.928426866, -23.9607857282,
                710.566564942, 622.850520903, 708.823861199, 239.271709191, 107.920489484, 29.0941796898]
weightsHidden = [238.856572902, 500.585619586, 236.012927652, 521.747953427, -194.288645044, -3.86396001374, -138.218661807, 260.884763267, 274.345688832, 344.580698035, 244.771265037, 335.459905805, 149.532537789, 411.724275878, 396.163485068, -15.8366335097, 71.7312996268, 219.718142062, 603.551599137, 451.761048128, 156.301116673, -163.698504422, 494.573946629, 186.034032158, 157.184064759, -192.219033522, 292.057636343, 247.082227442, 502.447217174, 346.165887865, 465.907541499, 228.293473372, 507.602566261, 354.857704211, 329.043432348, -110.145897036, 42.5344591278, 388.555657648, 252.605076497, 538.346015561, 316.819528571, 335.132907073, 66.6087874944, 448.3712095, 327.189396899, 110.753027962, 339.699111039, 459.717759735, 90.7523053115, 702.85715589, 740.194970594, -187.5146278, 222.579783893, -117.77366175, 92.6726260403, 466.965863166, 482.405553899, 139.229783621, -77.5020574273, 466.910865394, 601.360187604, 623.135448876, 755.530344977, 209.543202496]


for i in range(len(weightsConL)):
    print(i)
    w = weightsConL[i]
    nest.SetStatus([conn_l[i]], {"weight": w})

for i in range(len(weightsConR)):
    w = weightsConR[i]
    nest.SetStatus([conn_r[i]], {"weight": w})

for i in range(len(weightsConLH)):
    w = weightsConLH[i]
    nest.SetStatus([conn_lH[i]], {"weight": w})

for i in range(len(weightsConRH)):
    w = weightsConRH[i]
    nest.SetStatus([conn_rH[i]], {"weight": w})
    # print('rh weight ',i,' was set to ',w)

for i in range(len(weightsHidden)):
    w = weightsHidden[i]
    nest.SetStatus([conn_hidden[i]], {"weight": w})