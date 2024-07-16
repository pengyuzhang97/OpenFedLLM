import random
import numpy as np

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)
    np.random.seed(script_args.seed)  # Shuffle the dataset

    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    if fed_args.split_strategy == "non-iid":

        # Initial parameters

        mu, sigma = 100, 10  # Mean and standard deviation for generating raw probabilities

        num_clients = fed_args.num_clients

        total_samples = len(dataset)

        min_samples_per_client = 80  # Minimum number of samples guaranteed for each client

        # Generate raw probabilities

        raw_c_p = np.random.normal(mu, sigma, num_clients)

        # Clip and normalize to ensure valid probabilities, but before normalization, ensure a minimum probability for each client

        # First, ensure non-negativity

        raw_c_p = np.clip(raw_c_p, 0, None)

        # Allocate minimum samples to each client and calculate remaining samples for probabilistic distribution

        remaining_samples = total_samples - num_clients * min_samples_per_client

        adjusted_c_p = np.ones(num_clients) * min_samples_per_client / total_samples  # Base probability for minimum allocation

        # Distribute the remaining samples proportional to the raw probabilities, skipping clients who would receive less than min_samples_per_client if raw probabilities were used directly

        if remaining_samples > 0:
            # Normalize raw probabilities excluding the base allocation for minimum samples

            normalized_c_p_for_remaining = (raw_c_p - min_samples_per_client) / np.sum(raw_c_p - min_samples_per_client)

            # Avoid division by zero and ensure no negative probabilities

            normalized_c_p_for_remaining = np.clip(normalized_c_p_for_remaining, 0, 1)

            adjusted_c_p += normalized_c_p_for_remaining * remaining_samples / total_samples

        # Now adjusted_c_p sums up to 1 and guarantees at least min_samples_per_client for each client

        client_sample_sizes = np.random.multinomial(total_samples, adjusted_c_p)


        # # Generate probabilities that follow a normal distribution
        # mu, sigma = 10, 5  # Mean and standard deviation
        # c_p = np.random.normal(mu, sigma, fed_args.num_clients)
        #
        # # Ensure probabilities are non-negative
        # c_p = np.clip(c_p, 0, None)
        #
        # # Normalize probabilities so that their sum is 1
        # c_p /= np.sum(c_p)
        #
        # client_sample_sizes = np.random.multinomial(len(dataset), c_p)

        shuffled_indices = np.random.permutation(len(dataset))

        start_idx = 0
        for num_samples_per_client in client_sample_sizes:
            end_idx = start_idx + num_samples_per_client
            local_datasets.append(dataset.select(shuffled_indices[start_idx:end_idx]))
            start_idx = end_idx
    
    return local_datasets


# seems that this function can be combined with a data selection mechanism?
# i'm going to fix the number of local data
def get_dataset_this_round(dataset, round, fed_args, script_args):

    if script_args.max_steps == -1:
        return dataset
    else:
        num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
        num2sample = min(num2sample, len(dataset))
        random.seed(round)
        random_idx = random.sample(range(0, len(dataset)), num2sample)
        dataset_this_round = dataset.select(random_idx)

        # return dataset_this_round
        return dataset_this_round


# def get_dataset_this_round(dataset, round, fed_args, script_args):
#     num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
#     num2sample = min(num2sample, len(dataset))
#     random.seed(round)
#     random_idx = random.sample(range(0, len(dataset)), num2sample)
#     dataset_this_round = dataset.select(random_idx)
#
#     # return all dataset no matter what value the max_step has.
#     return dataset