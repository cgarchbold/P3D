import torch

def calculate_sigma(epsilon, n_critic, iterations, samples, privacy_violation):

    q = 1 / samples

    sqt = torch.sqrt(n_critic * iterations * torch.log(torch.tensor(1/privacy_violation)))

    return 2 * q * sqt / epsilon
    
    # the budget here should change with the total number of iterations.
    # https://mukulrathi.com/privacy-preserving-machine-learning/deep-learning-differential-privacy/
    # https://arxiv.org/pdf/1802.06739.pdf