import math
import pdb

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, need_skip=False, bias=False):
        super(GraphConvolution, self).__init__()
        # inout 166->16->2
        self.in_features = in_features
        self.out_features = out_features
        self.need_skip = need_skip
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(332, 2))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, origin_features):
        # pdb.set_trace()

        support = torch.mm(input, self.weight) # 128x166 * 166x16 -> 128x16
        output = torch.spmm(adj, support) # 128x128 * 128x16 -> 128x16
        if self.need_skip:
            support2 = torch.mm(origin_features, self.weight2)
            output = output + torch.spmm(adj, support2)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'