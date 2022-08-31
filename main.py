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
    show_data_plot = False  # Boolean, set to True to visualize data scatter diagram

    # !!!----- Student and teacher network choices -----!!!
    M = [500]       # number of student neurons e.g., [50,200,500]
    student_type = 'Tanh'
    bias_status_list = [True, False]
    outer_train_list = [True, False]

    # !!!----- Optimizater choices -----!!!
    num_epochs = 12000              # Number of epochs
    plot_incs = np.concatenate((np.arange(0, 10, 1), np.arange(10, 100, 10), np.arange(100, num_epochs + 1, 100))) # Points when to compute test error etc.
    rtypes = ['none', 'fro', 'nuc']               # Choose subset of ['none', 'fro', 'nuc']
    rweights = [0.1, 0.01, 0.001]   # Regularization weights to test
    step_size = 0.01

    # !!!----- Outcome thresholds -----!!!
    # Factors for classify different tests into categories of benign vs. non-benign overfit, fit or underfit.
    benign_ub = 1.2
    overfit_ub = 0.7
    fit_ub = 1.2


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
                            # Define data distribution
                            num_clusters = 2 # define number of clusters
                            mu = torch.cat((3*torch.ones(2), torch.zeros(d-2)), 0)
                            mu = [mu, -mu] # make a list of mean vectors one for each cluster
                            Sigma = torch.eye(d) # Define covariance matrix for each cluster
                            train_x, train_y, test_x, test_y, num_corrupted = df.generate_data(n, d, beta, teacher, data_dist, mu=mu, Sigma=Sigma, num_clusters=num_clusters)
                        else:
                            print("data_dist_not_recognized")

                        # Plot to check data looks correct
                        if show_data_plot:
                            plot_x = train_x.numpy()
                            plot_y = train_y.numpy().reshape(n)
                            sns.scatterplot(plot_x[:,0], plot_x[:,1], hue=plot_y)
                            plt.legend()
                            plt.show()

                        for bias_status in bias_status_list:
                            for outer_train in outer_train_list:
                                for l in range(len(rtypes)):
                                    if rtypes[l] == 'none':
                                        num_rweights = 1
                                    else:
                                        num_rweights = len(rweights)
                                    for k in range(num_rweights):
                                        loss_fn = mod.MSE_regularized(rtypes[l], rweights[k])
                                        if rtypes[l] == 'none':
                                            log_path = tb_path + data_dist + "_nClusters-" + str(num_clusters) + "_trainOuter-" + str(outer_train) + "_act-" + student_type + "_bias-" + str(bias_status) + "_n"   + str(n) + "_m" + str(m) + "_d" + str(d) + "_r" + str(r) + "_beta" + str(
                                                beta) + "_regtype-" + rtypes[l]
                                        else:
                                            log_path = tb_path + data_dist + "_nClusters-" + str(num_clusters) + "_trainOuter-" + str(outer_train) + "_act-" + student_type + "_bias-" + str(bias_status) + "_n" + str(n) + "_m" + str(m) + "_d" + str(
                                                d) + "_d" + str(r) + "_beta" + str(
                                                beta) + "_regtype-" + rtypes[l] + "_rweight-" + str(rweights[k])
                                        writer = SummaryWriter(log_path)

                                        # # !!!----- Define student neuron and optimizer ------!!!
                                        student = mod.One_Hidden_Layer_Model_Tanh(d, m, bias_status, outer_train)
                                        optimizer = torch.optim.SGD(student.parameters(), step_size)

                                        student, final_test, final_train = f.train_model(writer, student, optimizer, loss_fn, train_x, train_y, test_x, test_y, num_epochs,
                                                       num_corrupted, plot_incs, n, n, np.reshape(((w_star).detach()).numpy(), (d)))
                                        potential_bo = "unlikely"
                                        print("Training complete for following experiment run:")
                                        print(log_path)
                                        print("Final train error: " + str(final_train))
                                        print("Final test error: " + str(final_test))

                                        if final_test <= benign_ub*beta:
                                            test_outcome = "benign"
                                        else:
                                            test_outcome = "non-benign"

                                        if final_train <= overfit_ub*beta:
                                            training_outcome = "overfit"
                                        elif final_train > overfit_ub*beta and final_train<=fit_ub*beta:
                                            training_outcome = "fit"
                                        else:
                                            training_outcome = "underfit"

                                        print("OUTCOME: " + test_outcome + " " + training_outcome)




