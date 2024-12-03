import torch

def calculate_sigma(epsilon, n_critic, iterations, samples, privacy_violation):

    q = 1 / samples

    sqt = torch.sqrt(n_critic * iterations * torch.log(torch.tensor(1/privacy_violation)))

    return 2 * q * sqt / epsilon