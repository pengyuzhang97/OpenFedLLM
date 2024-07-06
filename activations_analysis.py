import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  gaussian_kde

import json
from pprint import  pprint

from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm

from safetensors.torch import load_file

import torch

data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r2a22_20240524225048/'

global_act_str = 'activations-global-'

local_act_str = 'activations-4-2'    # 2 is the round id

# global activation get based on data_path

def get_activations(data_path, round_id, glo=False):
    # get the activation
    if glo:
        act_file = data_path  + global_act_str + str(round_id) + '.pt'
    else:
        act_file = data_path  + local_act_str +'.pt'

    # safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge

    act = torch.load(act_file)


    # global_act = load_file(global_act_file)

    return act



layer_str = 'layers.0'
module_str = ['q_proj.lora_A', 'q_proj.lora_B', 'k_proj.lora_A', 'k_proj.lora_B', 'v_proj.lora_A',
              'v_proj.lora_B', 'o_proj.lora_A', 'o_proj.lora_B', 'up_proj.lora_A', 'up_proj.lora_B',
              'down_proj.lora_A', 'down_proj.lora_B', 'gate_proj.lora_A', 'gate_proj.lora_B']


act_ = get_activations(data_path, 1)

# print(global_act_round_1)


for i in range(len(module_str)):
    for k, v in act_.items():
        if layer_str in k and module_str[i] in k:

            # Got unsupported ScalarType BFloat16
            v = v.float()
            one_data = v[0, :, :].cpu().detach().numpy()

            one_data = np.absolute(one_data)

            # one_data[250, :] += 10

            # one_data is a n * m matrix, n is the sequence length, m is the hidden size
            # plot one_data in a 3d plot, n is the x axis, m is the y axis, on_data is the z axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x = np.arange(one_data.shape[1])
            y = np.arange(one_data.shape[0])
            x, y = np.meshgrid(x, y)
            z = one_data

            ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=10, antialiased=False)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Token')
            ax.set_zlabel('Absolute Value')
            ax.set_title(f'{k}')

            plt.show()



print('DONE')

        # print(k, v.shape)
