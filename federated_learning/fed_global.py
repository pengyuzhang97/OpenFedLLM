import copy
import random
import torch
import numpy as np







def Q_Deq_SymQ(input, num_bits, my_per_channel=False):

    if not my_per_channel:
        max_input = (
            torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
            .expand_as(input)
            .detach()
        )
    else:

        max_input = (
            torch.max(torch.abs(input), dim=0, keepdim=True)[0]
            .expand_as(input)
            .detach()
        )

    s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

    # q_output = torch.round(input * s)

    output = torch.round(input * s).div(s + 1e-6)

    return output



# def DeSymQ(input, num_bits):
#
#     max_input = (
#         torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
#         .expand_as(input)
#         .detach()
#     )
#
#     s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
#
#     # q_output = torch.round(input * s)
#
#     output = input.div(s + 1e-6)
#
#     return output



class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

        # q_output = torch.round(input * s)

        output = torch.round(input * s).div(s + 1e-6)



        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped inputlan
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None



def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate( script_args, fed_args, global_dict, local_dict_list,
                     sample_num_list, clients_this_round, round_idx,
                     proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, overall_drop_rate=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    else:   # Normal dataset-size-based aggregation

        # print('Start dropout and rescale')
        # print('Dropout on A')
        # for client in clients_this_round:
        #     for k, v in local_dict_list[client].items():
        #         # print('Dropout on ', k)
        #         if 'lora_A' in k:
        #             # print('Dropout on ', k)
        #             rescaled_mask = generate_scaled_binary_mask(v, overall_drop_rate)
        #             v.data *= rescaled_mask.to(v.device)
        #             del rescaled_mask
        # print('Dropout and rescale Done')

        # do i need to cancel out the arithmatic addtion?


        # for client in clients_this_round:
        #     with torch.no_grad():
        #         flat_weights = torch.nn.utils.parameters_to_vector(local_dict_list[client].values())
        #         rescaled_mask = generate_scaled_binary_mask_for_vector(flat_weights, overall_drop_rate)
        #         flat_weights.data *= rescaled_mask.to(flat_weights.device)
        #         del rescaled_mask
        #
        #         # new_weights = torch.nn.utils.vector_to_parameters(flat_weights, local_dict_list[client].values())
        #
        #         torch.nn.utils.vector_to_parameters(flat_weights, local_dict_list[client].values())
        #
        #         # for p, w in zip(local_dict_list[client].values(), flat_weights):
        #         #     p.copy_(w)


        # for key in global_dict.keys():
        #     global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        #
        # real_global_dict = copy.deepcopy(global_dict)

        # clients_this_round
        #
        # temp_for_0 = []
        # temp_for_1 = []
        # for k in local_dict_list[clients_this_round[0]].keys():
        #     if 'lora_A' in k:
        #         bk = k.replace('lora_A', 'lora_B')
        #
        #         temp0 = local_dict_list[clients_this_round[0]][bk].data @ local_dict_list[clients_this_round[0]][k].data
        #         norm_2 = torch.norm(temp0, p=2)
        #         temp_for_0.append(norm_2)
        #
        #         temp1 = local_dict_list[clients_this_round[1]][bk].data @ local_dict_list[clients_this_round[1]][k].data
        #         norm_2 = torch.norm(temp1, p=2)
        #         temp_for_1.append(norm_2)
        #
        #
        # numpy_list_0 = [tensor.cpu().numpy() for tensor in temp_for_0]
        # numpy_list_1 = [tensor.cpu().numpy() for tensor in temp_for_1]


        if script_args.quantize:
            # local_to_server
            for client in clients_this_round:
                for k, v in local_dict_list[client].items():
                    if 'lora_A' in k:
                        v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)
                    else:
                        v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)

                    # rescaled_mask = generate_scaled_binary_mask(v, overall_drop_rate)
                    # v.data *= rescaled_mask.to(v.device)
                    # del rescaled_mask
            print('Client-to-Global SymQ and DeSymQ are Done')


        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])


        # # server to local
        # for k, v in global_dict.items():
        #     if 'lora_A' in k:
        #         v.data = Q_Deq_SymQ(v, num_bits=4)
        #     else:
        #         v.data = Q_Deq_SymQ(v, num_bits=4)
        # print('Global-to-Client SymQ and DeSymQ are Done')


    return global_dict, global_auxiliary


def global_aggregate_hybrid(script_args, fed_args, hetero_local_dict_list,
                     sample_num_list, clients_this_round, overall_drop_rate=None, method='concat'):
    # method svd
    # method zero-padding

    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])

    # print('Start dropout and rescale')
    # print('Dropout on A')
    # for client in clients_this_round:
    #     for k, v in local_dict_list[client].items():
    #         # print('Dropout on ', k)
    #         if 'lora_A' in k:
    #             # print('Dropout on ', k)
    #             rescaled_mask = generate_scaled_binary_mask(v, overall_drop_rate)
    #             v.data *= rescaled_mask.to(v.device)
    #             del rescaled_mask
    # print('Dropout and rescale Done')

    # do i need to cancel out the arithmatic addtion?

    # for client in clients_this_round:
    #     with torch.no_grad():
    #         flat_weights = torch.nn.utils.parameters_to_vector(local_dict_list[client].values())
    #         rescaled_mask = generate_scaled_binary_mask_for_vector(flat_weights, overall_drop_rate)
    #         flat_weights.data *= rescaled_mask.to(flat_weights.device)
    #         del rescaled_mask
    #
    #         # new_weights = torch.nn.utils.vector_to_parameters(flat_weights, local_dict_list[client].values())
    #
    #         torch.nn.utils.vector_to_parameters(flat_weights, local_dict_list[client].values())
    #
    #         # for p, w in zip(local_dict_list[client].values(), flat_weights):
    #         #     p.copy_(w)

    # for key in global_dict.keys():
    #     global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    #
    # real_global_dict = copy.deepcopy(global_dict)

    # local_to_server
    if script_args.quantize:
        for client in clients_this_round:
            for k, v in hetero_local_dict_list[client].items():
                if 'lora_A' in k:
                    v.data = Q_Deq_SymQ(v, num_bits=8)
                else:
                    v.data = Q_Deq_SymQ(v, num_bits=8)

                # rescaled_mask = generate_scaled_binary_mask(v, overall_drop_rate)
                # v.data *= rescaled_mask.to(v.device)
                # del rescaled_mask
        print('Client-to-Global SymQ and DeSymQ are Done')

    global_dict = copy.deepcopy(hetero_local_dict_list[clients_this_round[0]])

    # method SVD
    if method == 'concat':
        for key in global_dict.keys():
            # concatenate in the loop
            id = 0
            for client in clients_this_round:
                if id == 0:
                    global_dict[key] = hetero_local_dict_list[client][key] * np.sqrt (sample_num_list[client] / sample_this_round)
                    id += 1
                else:
                    if global_dict[key].shape[0] < global_dict[key].shape[1]:
                        cat_dim = 0
                    else:
                        cat_dim = 1
                    global_dict[key] = torch.cat( (global_dict[key],
                                                 hetero_local_dict_list[client][key] * np.sqrt(sample_num_list[client] / sample_this_round)), dim=cat_dim)

    # TODO Quantization for dispatching should not be done here
    # quantization
    # # server to local
    # for k, v in global_dict.items():
    #     if 'lora_A' in k:
    #         v.data = Q_Deq_SymQ(v, num_bits=8)
    #     else:
    #         v.data = Q_Deq_SymQ(v, num_bits=8)
    # print('Global-to-Client SymQ and DeSymQ are Done')

    return global_dict





# P is the probability of true
# overall_drop_rate = 0.8
#
# avg_drop_rate = np.sqrt(overall_drop_rate)
#
# P = 1-avg_drop_rate
# def generate_scaled_binary_mask(layer_weight, P):
#     n, m = layer_weight.shape
#     mask = torch.bernoulli(torch.full((n, m), P)).bool()
#
#     rescaled_mask = mask * (1 / P)
#
#     return rescaled_mask


def generate_scaled_binary_mask(layer_weight, drop_rate):

    # avg_drop_rate = np.sqrt(drop_rate)
    # P = 1 - avg_drop_rate

    P = 1 - drop_rate

    n, m = layer_weight.shape
    mask = torch.bernoulli(torch.full((n, m), P)).bool()

    rescaled_mask = mask * (1 / drop_rate)


    return rescaled_mask



# def multi_stage_bernoulli_sampling(layer_weight, drop_rate, t):
#     P = 1 - drop_rate
#     n, m = layer_weight.shape
#
#     mask = torch.bernoulli(torch.full((n, m), P)).bool()
#



def generate_scaled_binary_mask_for_vector(layer_weight, drop_rate):

    # avg_drop_rate = np.sqrt(drop_rate)
    # P = 1 - avg_drop_rate

    P = 1 - drop_rate

    n = layer_weight.shape
    mask = torch.bernoulli(torch.full(n, P)).bool()

    rescaled_mask = mask * (1 / drop_rate)


    return rescaled_mask


if __name__ == "__main__":


    w_bits = 8

    # real_weights = torch.full((3, 4), 3, dtype=torch.float32)

    real_weights = torch.rand((3,4), dtype=torch.float32)

    quantizer = SymQuantizer

    weight_clip_val = torch.tensor([-2.0, 2.0])
    q_dq_weight = SymQuantizer.apply(
        real_weights, weight_clip_val, w_bits, False
    )


    # drop_rate = 0.1
    #
    # rescaled_mask = generate_scaled_binary_mask(weight, drop_rate)
    #
    # new_weight = weight * rescaled_mask
    #
    # print(new_weight)




    print('done')
