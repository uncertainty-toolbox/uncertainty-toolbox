import sys, math
from argparse import Namespace
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, use_bn=True,
                 actv_type='relu'):
        super(LinearLayer, self).__init__()

        """ linear layer """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif actv_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError

    def reset_parameters(self):
        # # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        # if (reset_indv_bias is None) or (reset_indv_bias is False):
        #     init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        # if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
        #     init.constant_(self.bias, 0)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # concat channels and length of signal if input is from conv layer
        if len(input.shape) > 2:
            batch_size = input.shape[0]
            input = input.view(batch_size, -1)

        out = F.linear(input, self.weight, self.bias)
        #print('after matmul\n', out)

        if self.bn:
            out = self.bn(out)
        #print('after bn\n', out)
        if self.activation is not None:
            out = self.activation(out)
        #print('after linear layer\n', out)

        return out


class pnn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_size=400, num_layers=4,
                 use_bn=False, actv_type='relu'):

        super(pnn, self).__init__()

        # create the mean network
        self.mean_fcs = nn.ModuleList()
        self.mean_fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.mean_fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.mean_fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))

        # create the variance network
        self.var_fcs = nn.ModuleList()
        self.var_fcs.append(LinearLayer(input_size, hidden_size, bias,
                                         use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers - 1):
            self.var_fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                             use_bn=use_bn, actv_type=actv_type))
        self.var_fcs.append(LinearLayer(hidden_size, output_size**2, bias,
                                         use_bn=False, actv_type=None))

        self.min_var_eps = 1e-8

    # def loss(self, pred_mean, pred_var, batch_y):
    #     diff = torch.sub(batch_y, pred_mean)
    #     for v in pred_var:
    #         if v == float('inf'):
    #             raise ValueError('infinite variance')
    #     loss = torch.mean(torch.div(diff**2, 2*pred_var))
    #     loss += torch.mean(torch.log(pred_var)/2)
    #
    #     return loss

    def forward(self, X):
        mean_X = X
        var_X = deepcopy(X)

        for layer in self.mean_fcs:
            mean_X = layer(mean_X)

        for layer in self.var_fcs:
            var_X = layer(var_X)

        mean_out = mean_X
        var_out = F.softplus(var_X) + self.min_var_eps

        return mean_out, var_out

    def loss(self, X, y):
        pred_mean, pred_var = self.forward(X)
        diff = torch.sub(y, pred_mean)
        for v in pred_var:
            if v == float('inf'):
                raise ValueError('infinite variance')
        loss = torch.mean(torch.div(diff**2, 2*pred_var))
        loss += torch.mean(torch.log(pred_var)/2)

        return loss
