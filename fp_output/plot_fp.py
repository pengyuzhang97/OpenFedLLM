import numpy as np
import matplotlib.pyplot as plt


fp_2m1mixed = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240622223939/global_loss.npy')



# only_att_hyb = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240626215320/global_loss.npy')


only_att_hyb = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240629123636/global_loss.npy')



# llama2-7b-chat
# lr = 1.5e-4
no_rewrite_bp= np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b16a1_l512_r32a64_20240706104223/global_loss.npy')




# print(only_att_hyb)


# x_all = np.arange(0, 200, 8)
# print(x_all)

st = 0
ra = 100


plt.figure()

# plt.plot( np.exp(fp_2m1mixed[st:ra]),label='fp_2m1mixed')
#
# plt.plot(np.exp(only_att_hyb), label = 'only_att_hyb')

plt.plot(np.exp(no_rewrite_bp), label = 'no_rewrite_bp_1.5e-4')




plt.plot()


# print(med_4bit_r8a45_data[st:ra])

plt.legend()
plt.show()