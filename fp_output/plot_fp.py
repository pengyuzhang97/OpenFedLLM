import numpy as np
import matplotlib.pyplot as plt





# fp_2m1mixed = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240622223939/global_loss.npy')



# only_att_hyb = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240626215320/global_loss.npy')


# only_att_hyb = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r8a16_20240629123636/global_loss.npy')



# llama2-7b-chat
# lr = 1.5e-4, bs=16, r32a64, has cuda ood if rewrite with bs=8
# no_rewrite_bp= np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b16a1_l512_r32a64_20240706104223/global_loss.npy')



# data count iid 1111111111111111111111
no_rewrte = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240707104704/global_loss.npy')
# 0.1 rewrite
rewrite_bp_top_1m1 = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240706222537/global_loss.npy')
rewrite_bp_down_1m1 = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240711120341/global_loss.npy')

# 100 round, iid
# r100_no_rewrite_bp = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240712172633/global_loss.npy')
r100_no_rewrite_bp_full = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240714222232/global_loss.npy')
r100_rewrite_bp_top_1m1_15m2 = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240712220233/global_loss.npy')
r100_rewrite_bp_down_1m1_5m1 = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240713130429/global_loss.npy')


r100_no_rewrite_bp_full_5p5e4lr = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240715143431/global_loss.npy')
r100_rewrite_bp_down_full_5p5e4lr = np.load('./CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240716095056/global_loss.npy')



# data count non-iid 1111111111111111111111
no_rewrte_noniid = np.load('./Non-iid_CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240709182159/global_loss.npy')
rewrite_bp_top_1m1_noniid = np.load('./Non-iid_CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240710152057/global_loss.npy')
rewrite_bp_down_1m1_noniid = np.load(
    'Non-iid_CodeAlpaca-20k_20000_fedavg_c20s2_i-1_b8a1_l512_r16a32_20240712102129/global_loss.npy')

# print(only_att_hyb)


# x_all = np.arange(0, 200, 8)
# print(x_all)

st = 0
ra = 10


plt.figure()

# plt.plot( np.exp(fp_2m1mixed[st:ra]),label='fp_2m1mixed')
#
# plt.plot(np.exp(only_att_hyb), label = 'only_att_hyb')

# plt.plot(np.exp(no_rewrite_bp), label = 'no_rewrite_bp_1.5e-4')


# 50 round
plt.plot(np.exp(no_rewrte[st:ra]), label = 'no_rewrte')
plt.plot(np.exp(rewrite_bp_top_1m1[st:ra]), label = 'rewrite_bp_top_1m1')
plt.plot(np.exp(rewrite_bp_down_1m1[st:ra]), label = 'rewrite_bp_down_1m1 ')



# # # 100 round
# # plt.plot(np.exp(r100_no_rewrite_bp[st:ra]), label = '100 round base')
# plt.plot(np.exp(r100_no_rewrite_bp_full[st:ra]), label = 'r100_no_rewrite_bp_full')
# plt.plot(np.exp(r100_rewrite_bp_top_1m1_15m2[st:ra]), label = 'r100_rewrite_bp_top_1m1_15m2')
# plt.plot(np.exp(r100_rewrite_bp_down_1m1_5m1), label = 'r100_rewrite_bp_down_1m1_5m1')
#
#
#
# plt.plot(np.exp(r100_no_rewrite_bp_full_5p5e4lr[st:ra]), label = 'r100_no_rewrite_bp_full_5e5lr')
# plt.plot(np.exp(r100_rewrite_bp_down_full_5p5e4lr[st:ra]), label = 'r100_rewrite_bp_down_full_5e5lr')


# # 100 round, non-iid
# plt.plot(np.exp(no_rewrte_noniid[st:ra]), label = 'no_rewrte_noniid')
# plt.plot(np.exp(rewrite_bp_top_1m1_noniid[st:ra]), label = 'rewrite_bp_top_1m1_noniid')
# plt.plot(np.exp(rewrite_bp_down_1m1_noniid[st:ra]), label = 'rewrite_bp_down_1m1_noniid ')



plt.plot()


# print(med_4bit_r8a45_data[st:ra])

plt.legend()
plt.show()