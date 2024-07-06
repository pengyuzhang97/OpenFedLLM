import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  gaussian_kde

import json
from pprint import  pprint

from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm

from safetensors.torch import load_file

import torch




############################################## quantization function ############################################
def Q_Deq_SymQ(input, num_bits):

    max_input = (
        torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
        .expand_as(input)
        .detach()
    )

    s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

    # q_output = torch.round(input * s)

    output = torch.round(input * s).div(s + 1e-6)

    return output



############################################## plot ############################################

g_list = []
keys = []


# data_path = './output/CodeAlpaca-20k_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240414134350/checkpoint-{}/adapter_model.safetensors'
#
# data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240410153935/checkpoint-{}/adapter_model.safetensors'

data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240419120253/checkpoint-{}/adapter_model.safetensors'


for i in range(20):

    g_data = load_file(data_path.format(i+1))

    if i == 0:
        for k in g_data:
            keys.append(k)

    g_list.append(g_data)


print('Load Done')


non_channel_Plot = True

plt.plot(0)

round_index = 5    # global model
layer_index = 1    # even (and 0) is A and odd is B

bits = 4
quant = False

# sum_up = True

if non_channel_Plot:

    data = g_list[round_index][keys[layer_index]]

    data = data.resize_(data.shape[0]*data.shape[1])

    if quant:
        input = Q_Deq_SymQ(data, num_bits=bits)
    else:
        input = data


    kde = gaussian_kde(input)
    dist_space = np.linspace(min(input), max(input), 100)
    plt.plot(dist_space, kde(dist_space))


    plt.xlabel('Value')
    plt.ylabel('Density')

    target = 'layer'
    title_index = keys[layer_index].index(target)

    if not quant:
        plt.title('Quant_None-Round_{}-{}'.format(round_index, keys[layer_index][title_index:]))
    else:
        plt.title('Quant_{}-Round_{}-{}'.format(bits, round_index, keys[layer_index][title_index:]))


    plt.show()


else:


    data = g_list[round_index][keys[layer_index]]
    per_channel_data = []

    channel_dim = min(data.shape)

    for i in range(channel_dim):

        if quant:
            input = Q_Deq_SymQ(data[:, i], num_bits=bits)
        else:
            input = data[:, i]

        # per_channel_data.append(data[:, i])

        per_channel_data.append(input)

    for j in range(channel_dim):
        kde = gaussian_kde(per_channel_data[j])
        dist_space = np.linspace(min(per_channel_data[j]), max(per_channel_data[j]), 100)
        plt.plot(dist_space, kde(dist_space))



    plt.xlabel('Value')
    plt.ylabel('Density')

    target = 'layer'
    title_index = keys[layer_index].index(target)
    if not quant:
        plt.title('Quant_None-Round_{}-{}'.format( round_index, keys[layer_index][title_index:]))
    else:
        plt.title('Quant_{}-PerChannel-Round_{}-{}'.format(bits, round_index, keys[layer_index][title_index:]))

    plt.show()




print('Plot Done')