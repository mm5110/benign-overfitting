import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def generate_data(n, beta, teacher, data_dist, **dist_statistics):
    num_corruptions = int(np.round(beta*n))
    if data_dist == 'gaussian':
        mu = dist_statistics['mu']
        Sigma = dist_statistics['Sigma']
        train_x, train_y = gaussian_data(n, num_corruptions, teacher, mu, Sigma)
        test_x, test_y = gaussian_data(n, num_corruptions, teacher, mu, Sigma)
    else:
        print("data_dist_not_recognized")

    return train_x, train_y, test_x, test_y, num_corruptions

def gaussian_data(n, num_corruptions, teacher, mu, Sigma):
    dist = MultivariateNormal(mu, Sigma)
    X = dist.sample((n,))             # returns tensor which is nxd
    y_clean = torch.sign(teacher(X))
    corruptions = torch.ones(n)
    corruptions[:num_corruptions] = -1
    y = y_clean*corruptions
    return X, y

def two_gaussian_data(d, n, beta, f_star, mu, Sigma):
    pass
