import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  gaussian_kde

import json
from pprint import  pprint

from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm

from safetensors.torch import load_file

import torch




############################################## plot ############################################

g_list = []
keys = []
# we track the final g computed by the Adam optimizer
weight_diff_norm = []
round = 200


# data_path = './output/CodeAlpaca-20k_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240414134350/checkpoint-{}/adapter_model.safetensors'
#
# data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240410153935/checkpoint-{}/adapter_model.safetensors'

# data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i20_b8a2_l512_r8a16_20240421232819/checkpoint-{}/adapter_model.safetensors'

data_path = './output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i100_b8a2_l512_r8a16_20240422095554/checkpoint-{}/adapter_model.safetensors'


for i in range(round):

    g_data = load_file(data_path.format(i+1))

    if i == 0:
        for k in g_data:
            keys.append(k)

    g_list.append(g_data)


# for i in range(len(g_list)):
#     for i in range(len(g_list[i])):



module_wise_weight_diff_norm = np.zeros((round-1, len(keys)))

# this is the  whole weight difference norm
for i in range(1, round):
    param_diff_list = []
    for k in keys:
        # print(k)
        param_diff = g_list[i][k] - g_list[i-1][k]

        module_wise_weight_diff_norm[i-1][keys.index(k)] = torch.norm(param_diff, p=2)

        param_diff_list.append(param_diff.flatten())
    # compute the tensor l2 norm of the param_diff_list
    inter = torch.cat(param_diff_list)
    inter_norm = torch.norm(inter, p=2)
    weight_diff_norm.append(inter_norm)



# module_index = 0
# plt.figure()
# # show the title based on the key given module_index
# plt.title(keys[module_index])
# # show the xlabel
# plt.xlabel('Round')
# # show the ylabel
# plt.ylabel('Weight Difference Norm')
# # plot the weight_diff_norm
# plt.plot(module_wise_weight_diff_norm[:, module_index])
# plt.show()








# plot the weight_diff_norm, add title, xlabel, ylabel
plt.figure()
plt.title('Weight Difference Norm')
plt.xlabel('Round')
plt.ylabel('Weight Difference Norm')

# plot the weight_diff_norm
plt.plot(weight_diff_norm)
plt.show()

