# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from matplotlib import pyplot as plt
import numpy as np
import functions as func
import models as mod
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
import data_functions as df
import functions as f

if __name__ == '__main__':

    # !!!----- Path to save down tensorboard logs -----!!!
    tb_path = "bo_tb_logs/"
    im_path = "bo_images/"

    # !!!----- Data distribution choices for features x -----!!!
    # For each run choose just one data distribution from ['gaussian', 'uniform', '2gaussian']
    # gaussian: symmetric gaussian with mean mu and covariance Sigma
    data_dist = 'gaussian'
    D = [300]       # ambient data dimension e.g., [50,200,500]
    R = [2]           # 'signal' dimension (classification depends only on the first r dimensions)
    Betas = [0.05]     # proportion of corruptions, e.g., [0.01, 0.05, 0.1, 0.2]
    N = [500]       # number of data points

    # !!!----- Student and teacher network choices -----!!!
    M = [500]       # number of student neurons e.g., [50,200,500]
    student_type = 'Tanh'
    bias_status = True

    # !!!----- Optimizater choices -----!!!
    num_epochs = 6000              # Number of epochgit s
    plot_incs = np.concatenate((np.arange(0, 10, 1), np.arange(10, 100, 10), np.arange(100, num_epochs + 1, 100))) # Points when to compute test error etc.
    rtypes = ['none']               # Choose subset of ['none', 'fro', 'nuc']
    rweights = [0.1, 0.01, 0.001]   # Regularization weights to test
    step_size = 0.01

    for n in N:
        for m in M:
            for d in D:
                for r in R:
                    # !!!----- Define teacher network ------!!!
                    w_star = -(1 / np.sqrt(2)) * torch.cat((torch.ones(r), torch.zeros(d - r)))
                    w_star = torch.reshape(w_star, (1, d))
                    teacher = mod.One_Hidden_Layer_Model_Tanh(d, 1, bias_status = False)
                    teacher.fc1.weight = nn.Parameter(w_star, requires_grad=False)
                    for beta in Betas:
                        if data_dist == 'gaussian':
                            mu = torch.zeros(d) #torch.cat((torch.ones(r), torch.zeros(d - r)))  # Mean vector
                            Sigma = torch.eye(d)                                # Covariance (prob)
                            train_x, train_y, test_x, test_y, num_corrupted = df.generate_data(n, beta, teacher, data_dist, mu=mu, Sigma=Sigma)
                        else:
                            print("data_dist_not_recognized")
                        for l in range(len(rtypes)):
                            if rtypes[l] == 'none':
                                num_rweights = 1
                            else:
                                num_rweights = len(rweights)
                            for k in range(num_rweights):
                                loss_fn = mod.MSE_regularized(rtypes[l], rweights[k])
                                if rtypes[l] == 'none':
                                    log_path = tb_path + data_dist + "_act-" + student_type + "_bias-" + str(bias_status) + "_n"   + str(n) + "_m" + str(m) + "_d" + str(d) + "_d" + str(r) + "_beta" + str(
                                        beta) + "_regtype-" + rtypes[l]
                                else:
                                    log_path = tb_path + data_dist + data_dist + "_act-" + student_type + "_bias-" + str(bias_status) + "_n" + str(n) + "_m" + str(m) + "_d" + str(
                                        d) + "_d" + str(r) + "_beta" + str(
                                        beta) + "_regtype-" + rtypes[l] + str(rweights[k])
                                writer = SummaryWriter(log_path)

                                # # !!!----- Define student neuron and optimizer ------!!!
                                student = mod.One_Hidden_Layer_Model_Tanh(d, m, bias_status)
                                optimizer = torch.optim.SGD(student.parameters(), step_size)

                                f.train_model(writer, student, optimizer, loss_fn, train_x, train_y, test_x, test_y, num_epochs,
                                               num_corrupted, plot_incs, n, n, np.reshape(((w_star).detach()).numpy(), (d)))

