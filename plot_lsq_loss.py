import numpy as np
import matplotlib.pyplot as plt



med_base_lsq4_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240411172116/global_loss.npy')


st = 0
ra = 15


plt.figure()

plt.plot( med_base_lsq4_data[st:ra],label='med_base_lsq4_data ')

plt.legend()
plt.show()