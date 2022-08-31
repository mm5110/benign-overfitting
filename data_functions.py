import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def generate_data(n, d, beta, teacher, data_dist, **dist_statistics):
    num_corruptions = 0
    if data_dist == 'gaussian':
        num_clusters = dist_statistics['num_clusters']
        Sigma = dist_statistics['Sigma']
        cluster_size = int(np.round(n/num_clusters))
        train_x = torch.empty((n,d))
        test_x = torch.empty((n,d))
        train_y = torch.empty((n,))
        test_y = torch.empty((n,))
        for i in range(num_clusters):
            mu = dist_statistics['mu'][i]
            train_x[i*cluster_size:(i+1)*cluster_size, :], train_y[i*cluster_size:(i+1)*cluster_size], train_corrups = gaussian_data(cluster_size, beta, teacher, mu, Sigma)
            test_x[i*cluster_size:(i+1)*cluster_size, :], test_y[i*cluster_size:(i+1)*cluster_size],_ = gaussian_data(cluster_size, beta, teacher, mu, Sigma)
            num_corruptions += train_corrups
    else:
        print("data_dist_not_recognized")

    return train_x, train_y, test_x, test_y, num_corruptions

def gaussian_data(n, beta, teacher, mu, Sigma):
    num_corruptions = int(np.round(beta * n))
    dist = MultivariateNormal(mu, Sigma)
    X = dist.sample((n,))             # returns tensor which is nxd
    y_clean = torch.sign(teacher(X))
    corruptions = torch.ones(n)
    corruptions[:num_corruptions] = -1
    y = y_clean*corruptions
    return X, y, num_corruptions

