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
    # 'e_rev_I': -70.0,
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
    # 'e_rev_I': -70.0,
    'v_thresh': -50.0,
    'v_reset': -65.0,
    'i_offset': 0.0,
    }


output_layer_Body = sim.Population(2, sim.IF_cond_alpha, OUTPUT_PARAMS)
output_layer_Head = sim.Population(2, sim.IF_cond_alpha,  OUTPUT_PARAMS)
input_dvs = sim.Population(10, sim.IF_cond_alpha, INPUT_PARAMS)
input_jointsHead = sim.Population(3*2, sim.IF_cond_alpha, INPUT_PARAMS)
hidden_layer_right = sim.Population(4, sim.IF_cond_alpha, HIDDEN_PARAMS)
hidden_layer_left = sim.Population(4, sim.IF_cond_alpha, HIDDEN_PARAMS)
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
                'Wmin': 10.,
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
output_layer = output_layer_Body  + output_layer_Head



input_Head = input_dvs  # + input_jointsHead
input_hidden = output_layer_Head + input_joints 
# input_Body = hidden_layer

circuit = output_layer + hidden_layer_right + hidden_layer_left + \
    input_dvs    + input_joints

syn_con_dvsHead = nest.Connect(map(int, input_Head.all_cells),
                             map(int, output_layer_Head.all_cells),
                               syn_spec={'model': 'syn_Ouput_Head', 'weight': {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})
                                # syn_spec = {'model': 'syn_Ouput', 'weight': {'distribution': 'normal_clipped', 'low': 1., 'high': 400., 'mu': 200.0, 'sigma': 200.0}, 'delay': 1.0})
#syn_spec = {'model': 'syn_Hidden', 'weight': 100., 'delay': 1.0})  # {'distribution': 'normal_clipped', 'low': 2., 'high': 400., 'mu': 50.0, 'sigma': 300.0}, 'delay': 1.0})
# temp = map(int, hidden_layer.all_cells)
# temp.reverse()
syn_con_inHidden = nest.Connect(map(int, input_hidden.all_cells),
                                map(int, hidden_layer_right.all_cells),
                                syn_spec={'model': 'syn_Hidden', 'weight':  {'distribution': 'uniform', 'low': 10., 'high': w0_max}, 'delay': 1.0})
syn_con_inHidden = nest.Connect(map(int, input_hidden.all_cells),
                                map(int, hidden_layer_left.all_cells),
                                syn_spec={'model': 'syn_Hidden', 'weight':  {'distribution': 'uniform', 'low': 10., 'high': w0_max}, 'delay': 1.0})
#    {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}
syn_con_right = nest.Connect(map(int, hidden_layer_right.all_cells),
                             map(int, output_layer_Body.all_cells[0:1]),
                             syn_spec={'model': 'syn_Ouput', 'weight':   {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})#{'distribution': 'normal', 'mu': 20300.0, 'sigma': 800.0}, 'delay': 1.0})
syn_con_left = nest.Connect(map(int, hidden_layer_left.all_cells),
                            map(int, output_layer_Body.all_cells[1:2]),
                             syn_spec={'model': 'syn_Ouput', 'weight':   {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})#{'distribution': 'normal', 'mu': 300.0, 'sigma': 200.0}, 'delay': 1.0})
#  syn_spec={'model': 'syn_Ouput', 'weight': {'distribution': 'normal', 'mu': 0.0, 'sigma': 300.0}, 'delay': 1.0})



hidden_layer = hidden_layer_right + hidden_layer_left

nest.Connect(map(int, output_layer),
             map(int, spike_detector), "one_to_one")

conn_r = nest.GetConnections(target=[output_layer[0]])
conn_l = nest.GetConnections(target=[output_layer[1]])

conn_rH = nest.GetConnections(target=[output_layer[2]])
conn_lH = nest.GetConnections(target=[output_layer[3]])



conn_hidden = nest.GetConnections(target=[hidden_layer[i] for i in range(len(hidden_layer))])

#copy weights array from csv in variable
weightsConL = [71.345438577, 131.250165395, 13.1929963054, 75.6823488976]
weightsConR = [33.8538790914, 32.5015568923, 49.4873286195, 15.1215874075]
weightsConLH = [120.306408842, 1710.83950924, 554.212192016, -581.442579182,
                815.944278823, 198.412137362, 925.328168068, 387.433829173, 18.7936138225, 172.312358539]
weightsConRH = [906.599431825, -809.58051259, 719.223139676, 1709.95899845, -
                170.340009361, 277.912100607, -526.740948738, 528.153009093, 980.834280407, 34.6624116253]
weightsHidden = [21.5562659427, 54.1790241036, 72.4199365569, -17.7073710075, -241.685084771, -607.892202259, -435.694485566, -613.262550301, -1142.58943917, -689.411219304, -849.986068538, -625.946872133, -349.338716759, 54.9323282495, -80.3085072459, 189.849028121, 571.761724758, 712.179733184, 199.009802682, 295.143623678, -40.0123426514, 104.795371384, -42.9349567819, -59.609823249, 210.249062668, 343.945338537, -56.0936288388, 153.169774055, 402.171551783, 212.188167409, 27.6766510544, 56.3207350895, 823.667750664, 410.548077197, 64.6974058667, 145.300921418, 408.117759719, 225.584634029, 366.847539289, 28.4589336457, 276.124811566, 48.0824625137, 311.455325945, 208.33953184, 213.422766068, 350.163053536, 340.756698329, -11.4971220342, -247.889584112, 14.0612367106, 242.540026057, 102.545361245, 103.361315644, 472.22618482, 773.823100897, -3.14754084688, 1481.72502386, 754.128567898, 1031.73235479, 1027.35548576, -184.908820766, -42.9996875706, 44.2305449087, 245.213954312]
#assign each connection corresponding weights
#1!check nest doko how to assign weights

for i in range(len(weightsConL)):
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
