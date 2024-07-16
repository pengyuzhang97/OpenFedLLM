import math

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage:
    num_rounds1 = 100
    initial_lr = 1.5e-5
    num_rounds2 = 50
    min_lr = 1e-6

    lr_list_1 = []
    lr_list_2 = []
    for round in range(num_rounds1):
        lr1 = cosine_learning_rate(round, num_rounds1, initial_lr, min_lr)
        lr_list_1.append(lr1)
        print(f"Round {round + 1}/{num_rounds1}, Learning Rate: {lr1:.8f}")

    for round in range(num_rounds2):
        lr2 = cosine_learning_rate(round, num_rounds2, initial_lr, min_lr)
        lr_list_2.append(lr2)
        print(f"Round {round + 1}/{num_rounds2}, Learning Rate: {lr2:.8f}")



    plt.plot(lr_list_1)
    plt.plot(lr_list_2)
    plt.show()
