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
input_dvs = sim.Population(10, sim.IF_cond_alpha, INPUT_PARAMS)

vtLeft = nest.Create('volume_transmitter', 1)
vtRight = nest.Create('volume_transmitter', 1)

spike_detector = nest.Create("spike_detector", 2, params={
    "withgid": True, "withtime": True})


############################
###### CONNECT LAYERS ######
############################

nest.CopyModel('stdp_dopamine_synapse', 'syn_Ouput_Head',
               {'Wmax': w_max,
                'Wmin': w_min,
                'A_plus': 0.1,
                'A_minus': -0.1,
                'tau_c': 40.,
                'tau_n': 20.,  # ,
                'b': 0.,
                   'vt': vtRight[0]
                })

output_layer = output_layer_Body 


input_Head = input_dvs

circuit = output_layer + input_dvs

syn_con_dvs = nest.Connect(map(int, input_Head.all_cells),
                               map(int, output_layer.all_cells),
                               syn_spec={'model': 'syn_Ouput_Head', 'weight': {'distribution': 'uniform', 'low': w0_min, 'high': w0_max}, 'delay': 1.0})

nest.Connect(map(int, output_layer),
             map(int, spike_detector), "one_to_one")

conn_r = nest.GetConnections(target=[output_layer[0]])
conn_l = nest.GetConnections(target=[output_layer[1]])

#copy weights array from csv in variable
weightsConL = [400.186831233, 2133.53446345, 521.212682388, -583.900182796, 862.990494895,
               657.52834364, 1509.77488163, -97.1861018306, 124.097782963, 526.386304751]
weightsConR = [625.83575506, -549.917791068, 605.593179725, 1895.08393048, 485.400743632,
               19.9331389972, -302.00386247, 407.549865978, 464.988836902, -23.2978259665]
#assign each connection corresponding weights
#1!check nest doko how to assign weights

# conn_hidden = conn_hidden_left + conn_hidden_right
for i in range(len(weightsConL)):
    w = weightsConL[i]
    nest.SetStatus([conn_l[i]], {"weight": w})

for i in range(len(weightsConR)):
    w = weightsConR[i]
    nest.SetStatus([conn_r[i]], {"weight": w})
