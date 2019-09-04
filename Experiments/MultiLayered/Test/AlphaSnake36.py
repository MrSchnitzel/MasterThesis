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
# weightsConL = [243.970100315, 161.174313432, -72.7432445096, 1422.13415872,
#                2030.34027183, 376.410172269, 552.636391496, 1207.85342302]
# weightsConR = [724.521678427, 1204.99912656, 1256.37712235,
#                69.1457515584, -1081.33182252, 778.496088725, 397.717444546, -176.761382489]
# weightsConLH = [-16.5413454845, 471.354009697, 53.3500979936, -24.3376294993,
#                 707.994624422, 622.855976917, 707.158651551, 237.708055047, 110.745422826, 29.1350283596]
# weightsConRH = [351.176149381, -591.983816347, 45.5098490773, 953.790832069,
#                 77.0395862389, -106.160076558, -483.295474873, 277.498127334, 785.178933014, 346.683638223]
# weightsHidden = [242.141901915,506.655696662,242.252247252,515.501005283,-195.281496451,-2.06377305512,-139.721644789,256.669492842,277.933081238,351.222437327,251.594484333,328.699848546,148.521158256,413.748705197,394.517591019,-20.2600221247,73.6344216518,223.387240699,607.69291723,448.362962923,156.182911479,-162.73869369,493.74812345,183.33161955,157.58185372,-191.771068954,292.45229733,246.216210736,501.292176409,346.454689816,465.641221213,227.614398057,509.528369024,358.549146738,333.19133121,-113.548371193,42.4253878975,389.531363239,251.777650759,535.610970406,317.21835978,335.580036023,67.0010517625,447.502453852,326.039230332,111.041881482,339.432636182,459.033189133,90.971081808,703.306587767,740.731498518,-187.901499502,222.615156927,-117.669915158,92.5683690024,466.627218732,482.764218899,139.83408048,-77.0631490325,466.112406011,600.96693237,623.374541203,755.363567474,209.18572647]
#assign each connection corresponding weights
#1!check nest doko how to assign weights
# weightsConR =
# weightsConRH =
# weightsConL =
# weightsConLH =
# weightsHidden=
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