import numpy as np
import matplotlib.pyplot as plt


med_8bit_r2a22_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r2a22_20240526150419/global_loss.npy')
med_4bit_r2a22_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r2a22_20240527122655/global_loss.npy')

med_8bit_r8a45_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r8a45_20240523223410/global_loss.npy')
med_4bit_r8a45_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r8a45_20240528095325/global_loss.npy')

med_4bit_r12a55_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c100s10_i10_b8a2_l512_r12a55_20240529121354/global_loss.npy')

# x_all = np.arange(0, 200, 8)
# print(x_all)

st = 0
ra = 100


plt.figure()

plt.plot( med_8bit_r2a22_data[st:ra],label='med_8bit_r2a22_data')
plt.plot( med_4bit_r2a22_data[st:ra],label='med_4bit_r2a22_data')


plt.plot(med_8bit_r8a45_data[st:ra], label = 'med_8bit_r8a45_data')
plt.plot(med_4bit_r8a45_data[st:ra], label = 'med_4bit_r8a45_data')

plt.plot(med_4bit_r12a55_data[st:ra], label = 'med_4bit_r12a55_data')

# print(med_4bit_r8a45_data[st:ra])

plt.legend()
plt.show()