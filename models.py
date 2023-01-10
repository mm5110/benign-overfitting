import torch
import torch.nn as nn
import numpy as np




# ----------- NEURAL NETWORKS ----------------
class TwoLayerNN(nn.Module):
    def __init__(self, d, m, bias_status, train_outer=False, act='ReLU'):
        super(TwoLayerNN, self).__init__()
        if act == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.fc1 = nn.Linear(d, m, bias=bias_status)
        self.fc2 = nn.Linear(m, 1, bias=False)
        output_weights = (1/m)*np.concatenate((-np.ones(int(m/2)), np.ones(int(m/2)))).reshape((1,m))
        output_weights = torch.Tensor(output_weights)
        if train_outer == False:
            self.fc2.weight = nn.Parameter(output_weights, requires_grad=False)
        else:
            self.fc2.weight = nn.Parameter(output_weights, requires_grad=True)

    def forward(self, x):
        y = self.fc1(x)
        z = self.activation(y)
        y = self.fc2(z)
        return y


# ----------- LOSS FUNCTIONS ----------------

class loss_regularized(nn.Module):
    """
    Class for constructing regularized MSE objective.
    """
    def __init__(self, loss_type, hidden_rtype, hidden_rweight, outer_rtype, outer_rweight):
        super(loss_regularized, self).__init__()
        self.loss_type = loss_type
        self.hidden_rtype = hidden_rtype
        self.outer_rtype = outer_rtype
        self.hidden_rweight = hidden_rweight
        self.outer_rweight = outer_rweight

    def forward(self, predictions, target, W, a):
        if self.loss_type == 'L2':
            error = torch.mean(torch.square(predictions - target))
        else:
            relu = nn.ReLU()
            error = torch.mean(relu(1 - predictions*target))
            # error = torch.mean(torch.maximum(torch.zeros_like(target), 1 - predictions*target)) # default is hinge
        if self.hidden_rtype == 'none':
            hidden_reg = 0
        else:
            hidden_reg = torch.linalg.matrix_norm(W, ord=self.hidden_rtype)
        if self.outer_rtype == 'none':
            outer_reg = 0
        else:
            outer_reg = torch.linalg.vector_norm(a, ord=self.outer_rtype)

        loss_value = error + self.hidden_rweight*hidden_reg + self.outer_rweight*outer_reg
        return loss_value



# ----------- OLD ----------------

# class One_Hidden_Layer_Model_ReLU(nn.Module):
#     """
#     Class for dense feedforward network with 1 hidden ReLU layer of width m and a fixed output layer with sigmoid output
#     non-linearity.
#     """
#     def __init__(self, d, m, bias_status):
#         super(One_Hidden_Layer_Model, self).__init__()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.fc1 = nn.Linear(d, m, bias=bias_status)
#         self.fc2 = nn.Linear(m, 1, bias=False)
#         a = (1/m)*np.concatenate((-np.ones(int(m/2)), np.ones(int(m/2)))).reshape((1,m))
#         output_weights = torch.Tensor(a)
#         self.fc2.weight = nn.Parameter(output_weights, requires_grad=False)
#
#     def forward(self, x):
#         y = self.fc1(x)
#         z = self.relu(y)
#         y = self.fc2(z)
#         y = 2*self.sigmoid(y) - 1
#         return y
#
#
# class One_Hidden_Layer_Model_Tanh(nn.Module):
#     """
#     Class for dense feedforward network with 1 hidden Tanh layer of width m, with fixed linear output layer.
#     """
#     def __init__(self, d, m, bias_status=False, train_outer=False):
#         super(One_Hidden_Layer_Model_Tanh, self).__init__()
#         self.tanh = nn.Tanh()
#         self.fc1 = nn.Linear(d, m, bias=bias_status)
#         self.fc2 = nn.Linear(m, 1, bias=False)
#         output_weights = (1 / m) * torch.ones(m)
#         if train_outer == False:
#             self.fc2.weight = nn.Parameter(output_weights, requires_grad=False)
#         else:
#             self.fc2.weight = nn.Parameter(output_weights, requires_grad=True)
#
#     def forward(self, x):
#         y = self.fc1(x)
#         z = self.tanh(y)
#         y = self.fc2(z)
#         return y
#
#
# class Two_Layer_Model_ReLU(nn.Module):
#     """
#     Class for dense feedforward network with 1 hidden ReLU layer of width m and univariate sigmoid output layer. Note both hidden
#     and output layer are trainable.
#     """
#
#     def __init__(self, d, m):
#         super(Two_Layer_Model, self).__init__()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#         self.fc1 = nn.Linear(d, m, bias=True)
#         self.fc2 = nn.Linear(m, 1, bias=False)
#
#     def forward(self, x):
#         y = self.fc1(x)
#         z = self.relu(y)
#         y = self.fc2(z)
#         y = 2*self.sigmoid(y) - 1
#
#         return y