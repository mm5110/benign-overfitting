import torch
import torch.nn as nn
import numpy as np




# ----------- NEURAL NETWORKS ----------------
class One_Hidden_Layer_Model_ReLU(nn.Module):
    """
    Class for dense feedforward network with 1 hidden ReLU layer of width m and a fixed output layer with sigmoid output
    non-linearity.
    """
    def __init__(self, d, m, bias_status):
        super(One_Hidden_Layer_Model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(d, m, bias=bias_status)
        self.fc2 = nn.Linear(m, 1, bias=False)
        a = (1/m)*np.concatenate((-np.ones(int(m/2)), np.ones(int(m/2)))).reshape((1,m))
        output_weights = torch.Tensor(a)
        self.fc2.weight = nn.Parameter(output_weights, requires_grad=False)

    def forward(self, x):
        y = self.fc1(x)
        z = self.relu(y)
        y = self.fc2(z)
        y = 2*self.sigmoid(y) - 1
        return y


class One_Hidden_Layer_Model_Tanh(nn.Module):
    """
    Class for dense feedforward network with 1 hidden Tanh layer of width m, with fixed linear output layer.
    """
    def __init__(self, d, m, bias_status=False, train_outer=False):
        super(One_Hidden_Layer_Model_Tanh, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(d, m, bias=bias_status)
        self.fc2 = nn.Linear(m, 1, bias=False)
        output_weights = (1 / m) * torch.ones(m)
        if train_outer == False:
            self.fc2.weight = nn.Parameter(output_weights, requires_grad=False)
        else:
            self.fc2.weight = nn.Parameter(output_weights, requires_grad=True)

    def forward(self, x):
        y = self.fc1(x)
        z = self.tanh(y)
        y = self.fc2(z)
        return y


class Two_Layer_Model_ReLU(nn.Module):
    """
    Class for dense feedforward network with 1 hidden ReLU layer of width m and univariate sigmoid output layer. Note both hidden
    and output layer are trainable.
    """

    def __init__(self, d, m):
        super(Two_Layer_Model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(d, m, bias=True)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        y = self.fc1(x)
        z = self.relu(y)
        y = self.fc2(z)
        y = 2*self.sigmoid(y) - 1

        return y


# ----------- LOSS FUNCTIONS ----------------

class MSE_regularized(nn.Module):
    """
    Class for constructing regularized MSE objective.
    """
    def __init__(self, rtype, rweight):
        super(MSE_regularized, self).__init__()
        self.rtype = rtype
        if self.rtype == 'none':
            self.rweight = 0
        else:
            self.rweight = rweight

    def forward(self, predictions, target, W):
        square_difference = torch.square(predictions - target)
        if self.rtype == 'none':
            reg = 0
        else:
            reg = torch.linalg.matrix_norm(W, ord=self.rtype)
        loss_value = torch.mean(square_difference) + self.rweight*reg
        return loss_value