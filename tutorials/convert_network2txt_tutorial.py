import os

from brian2 import PoissonGroup
from brian2 import StateMonitor, SpikeMonitor
from brian2 import ms

from teili import Neurons, Connections, TeiliNetwork
from teili.models.neuron_models import LinearLIF as neuron_model
from teili.models.synapse_models import Exponential as static_synapse_model

from speed.teili2orca import Speed

import numpy as np

def gen_conn_matrix(source_num_neu, target_num_neu, prob):
    conns_sample_space = source_num_neu * target_num_neu
    conns_samples = int(prob * conns_sample_space)
    conn_matrix = np.zeros(conns_sample_space)
    conn_matrix[rng.choice(range(conns_sample_space), conns_samples, replace=False)] = 1
    conn_matrix = np.reshape(conn_matrix, (source_num_neu, target_num_neu))
    sources, targets = conn_matrix.nonzero()

    return sources, targets

rng = np.random.default_rng(12345)

# Defining the network
N_p1 = 81#92
N_p2 = 20#48

Net = TeiliNetwork()

p1 = Neurons(N_p1, equation_builder=neuron_model(num_inputs=2),
    name='P1')
p2 = Neurons(N_p2, equation_builder=neuron_model(num_inputs=2),
    name='P2')

p1_p2 = Connections(p1, p2,
    equation_builder=static_synapse_model(), name='synapse_P1_P2')
p2_p1 = Connections(p2, p1,
    equation_builder=static_synapse_model(), name='synapse_P2_P1')
p1_p1 = Connections(p1, p1,
    equation_builder=static_synapse_model(), name='synapse_P1_P1')
p2_p2 = Connections(p2, p2,
    equation_builder=static_synapse_model(), name='synapse_P2_P2')

sources, targets = gen_conn_matrix(source_num_neu=N_p1, target_num_neu=N_p2, prob=.15)
p1_p2.connect(i=sources, j=targets)
sources, targets = gen_conn_matrix(source_num_neu=N_p2, target_num_neu=N_p1, prob=.85)
p2_p1.connect(i=sources, j=targets)
sources, targets = gen_conn_matrix(source_num_neu=N_p1, target_num_neu=N_p1, prob=.85)
p1_p1.connect(i=sources, j=targets)
sources, targets = gen_conn_matrix(source_num_neu=N_p2, target_num_neu=N_p2, prob=.10)
p2_p2.connect(i=sources, j=targets)

Net.add(p1, p2, p1_p2, p2_p1, p1_p1, p2_p2)

# Simulating the networ
Net.run(1*ms, report='text')

# Converting the network
converted_model = Speed(Net)

# Print the converted model properties
converted_model.print_network()

# TODO put it on utils.py and create tutorial with load and/or save
from collections import defaultdict
group_conns = defaultdict(list)
map_group_targets = defaultdict(list)
for key, val in converted_model.synapse_populations.items():
    group_conns[val[0]].append(val[1])
    map_group_targets[val[0]].append(key)

with open('orca_net.txt', 'w') as f:
    for key, val in group_conns.items():
        source = key
        num_neurons = converted_model.neuron_populations[source]
        targets = ','.join(val)
        probs = [str(converted_model.synapse_tags[k]['p_connection']) for k in map_group_targets[source]]
        probs = ','.join(probs)
        f.write('{}\t{}\t{}\t{}\n'.format(source, num_neurons, targets, probs))
