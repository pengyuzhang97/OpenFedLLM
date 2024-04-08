import numpy as np
import matplotlib.pyplot as plt



# med_base_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r32a64_20240305150901/global_loss.npy')

med_base_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240308225100/global_loss.npy')    # previous one

# drop_B_med_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240311143410/global_loss.npy')
#
#
# drop_A_med_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240311171050/global_loss.npy')
#
#
# drop_A_all_clients_med_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s20_i10_b8a2_l512_r8a16_20240312143845/global_loss.npy')


med_base_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240317201140/global_loss.npy')


# a and w = 8bits
med_8qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240317130944/global_loss.npy')

# a and w = 4bits
med_4qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240325174707/global_loss.npy')


med_4_nonloc_comm_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240331110901/global_loss.npy')

med_8_nonloc_comm_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240401114315/global_loss.npy')


# a and w = 4, comm = 8bits
med_4_8_hasloc_comm_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240401165139/global_loss.npy')

# a and w = 8, comm = 4bits
med_8_4_hasloc_comm_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240403100004/global_loss.npy')

# a and w, comm = 4bits
med_4_hasloc_comm_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240328164446/global_loss.npy')


med_4_hasloc_no_download_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240330135807/global_loss.npy')


med_4_hasloc_no_upload_qat_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240330210119/global_loss.npy')



med_4_new_no_comm_data = np.load('./output/medical_meadow_medical_flashcards_20000_fedavg_c20s2_i10_b8a2_l512_r8a16_20240330225237/global_loss.npy')



x_all = np.arange(0, 200, 8)
print(x_all)

st = 0
ra = 50


plt.figure()

plt.plot( med_base_data[st:ra],label='med_base')
# plt.plot(drop_B_med_data[:3],label='med_drop_B')
# plt.plot(drop_A_med_data[:5],label='med_drop_A')
# plt.plot(drop_A_all_clients_med_data[:],label='med_drop_A_all')

# plt.plot(med_8qat_data[:9], label='med_8qat')

plt.plot( med_4qat_data[st:ra], label='med_4qat')
# plt.plot(med_4_new_no_comm_data[:ra], label='med_4_new_comm_data')

plt.plot(med_4_nonloc_comm_qat_data[st:ra], label = 'med_4_nonloc_comm_qat')

plt.plot(med_8_nonloc_comm_qat_data[st:ra], label='med_8_nonloc_comm_qat')


plt.plot(med_4_8_hasloc_comm_qat_data[st:ra], label='med_4_8_hasloc_comm_qat_data')

plt.plot(med_8_4_hasloc_comm_qat_data[st:ra], label='med_8_4_hasloc_comm_qat_data')

# plt.plot(med_4_hasloc_comm_qat_data[st:ra], label='med_4_hasloc_comm_qat')
#
# plt.plot(med_4_hasloc_no_download_qat_data[st:ra], label='med_4_hasloc_no_down_qat')
#
# plt.plot(med_4_hasloc_no_upload_qat_data[st:ra], label='med_4_hasloc_no_up_qat')






# plt.plot(med_base_data[:22] - med_qat_data, label='diff')

plt.legend()
plt.show()