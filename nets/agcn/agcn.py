import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

from IPython import embed

import os


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)




class GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.gcn1(x) #+ self.residual(x)
        return self.relu(x)




class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)


        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)







class Model_quarterChannel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_quarterChannel, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64//4, A, residual=False)
        self.l2 = TCN_GCN_unit(64//4, 64//4, A)
        self.l3 = TCN_GCN_unit(64//4, 64//4, A)
        self.l4 = TCN_GCN_unit(64//4, 64//4, A)
        self.l5 = TCN_GCN_unit(64//4, 128//4, A, stride=2)
        self.l6 = TCN_GCN_unit(128//4, 128//4, A)
        self.l7 = TCN_GCN_unit(128//4, 128//4, A)
        self.l8 = TCN_GCN_unit(128//4, 256//4, A, stride=2)
        self.l9 = TCN_GCN_unit(256//4, 256//4, A)
        self.l10 = TCN_GCN_unit(256//4, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)




class Model_thin(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)

    def forward_featuremap(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        feat = x 
        nm2,c2,t2,v2=feat.shape
        feat = feat.reshape(N,M,c2,t2,v2)
        # (n,c,t)
        feat = feat.mean(4).mean(1)
        # (n,t,c)
        feat = feat.permute(0,2,1).contiguous()
        featuremap = feat


        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature, featuremap
        else:
            return self.fc(x)


    def get_IN(self, x):
        assert len(x.shape)==4
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        eps=1e-6
        sig = (var + eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        stats=torch.cat([mu,sig],1)
        stats=stats.squeeze()
        return stats

    def forward_all_feature_stats(self, x):

        # stats1=self.get_IN(x)
        # stats2=self.get_IN(x)
        # stats3=self.get_IN(x)
        # stats4=self.get_IN(x)
        # return x, [stats1, stats2, stats3, stats4]

        N, C, T, V, M = x.size()
        assert M==1
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        stats1=self.get_IN(x)
        x = self.l2(x)
        stats2=self.get_IN(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        stats3=self.get_IN(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        stats4=self.get_IN(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.fc(x)

        return x, [stats1, stats2, stats3, stats4]
        








class Model_thin_jigsaw(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
        n_jigsaw=6):
        super(Model_thin_jigsaw, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.n_jigsaw = n_jigsaw

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        self.fc_jigsaw = nn.Linear(256, self.n_jigsaw)
        nn.init.normal_(self.fc_jigsaw.weight, 0, math.sqrt(2. / self.n_jigsaw))

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), self.fc_jigsaw(x), feature
        else:
            return self.fc(x), self.fc_jigsaw(x)








class Model_thin_trecon(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)

    def forward_featuremap(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        feat = x 
        nm2,c2,t2,v2=feat.shape
        feat = feat.reshape(N,M,c2,t2,v2)
        # (n,c,t)
        feat = feat.mean(4).mean(1)
        # (n,t,c)
        feat = feat.permute(0,2,1).contiguous()
        featuremap = feat


        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature, featuremap
        else:
            return self.fc(x)


    def get_IN(self, x):
        assert len(x.shape)==4
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        eps=1e-6
        sig = (var + eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        stats=torch.cat([mu,sig],1)
        stats=stats.squeeze()
        return stats

    def forward_all_feature_stats(self, x):

        # stats1=self.get_IN(x)
        # stats2=self.get_IN(x)
        # stats3=self.get_IN(x)
        # stats4=self.get_IN(x)
        # return x, [stats1, stats2, stats3, stats4]

        N, C, T, V, M = x.size()
        assert M==1
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        stats1=self.get_IN(x)
        x = self.l2(x)
        stats2=self.get_IN(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        stats3=self.get_IN(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        stats4=self.get_IN(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.fc(x)

        return x, [stats1, stats2, stats3, stats4]
        








class Model_thin_CodtRecon(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin_CodtRecon, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        # self.fc_jigsaw = nn.Linear(256, self.n_jigsaw)
        # nn.init.normal_(self.fc_jigsaw.weight, 0, math.sqrt(2. / self.n_jigsaw))

        hidden_dim=256
        assert len(A.shape)==3
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_dim, in_channels * A.shape[1]) )

        # if graph_args['layout'] == 'skeletics':
        #     self.parts = [[2, 3, 4], [5, 6, 7], [0, 1, 8], [9, 10, 11], [12, 13, 14]]
        # elif graph_args['layout'] == 'ntu-rgb+d':
        #     self.parts = [[3 - 1, 4 - 1, 1 - 1, 2 - 1, 21 - 1],
        #                   [5 - 1, 6 - 1, 7 - 1, 8 - 1, 22 - 1, 23 - 1],
        #                   [9 - 1, 10 - 1, 11 - 1, 12 - 1, 24 - 1, 25 - 1],
        #                   [13 - 1, 14 - 1, 15 - 1, 16 - 1],
        #                   [17 - 1, 18 - 1, 19 - 1, 20 - 1]]
        # elif graph_args['layout'] == 'openpose':
        #     self.parts = [[2, 3, 4, 1],
        #                   [5, 6, 7, 1],
        #                   [8, 9, 10, 1],
        #                   [11, 12, 13, 1],
        #                   [0, 14, 15, 16, 17]]
        # elif graph_args['layout'] == 'skeletics25':
        #     self.parts = [[2, 3, 4, 1],
        #                   [5, 6, 7, 1],
        #                   [8, 9, 10, 11],
        #                   [8, 12, 13, 14],
        #                   [0, 15, 16, 17, 18]]
        self.parts = [[3 - 1, 4 - 1, 1 - 1, 2 - 1, 21 - 1],
                    [5 - 1, 6 - 1, 7 - 1, 8 - 1, 22 - 1, 23 - 1],
                    [9 - 1, 10 - 1, 11 - 1, 12 - 1, 24 - 1, 25 - 1],
                    [13 - 1, 14 - 1, 15 - 1, 16 - 1],
                    [17 - 1, 18 - 1, 19 - 1, 20 - 1]]

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        dec_x = x.mean(dim=-1)
        dec_x = dec_x.permute(0, 2, 1).contiguous()

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), self.decoder(dec_x), feature
        else:
            return self.fc(x), self.decoder(dec_x)

    def masking(self, im_q, mask_p):
        N, C, T, V, M = im_q.size()
        mask = torch.rand(N, 1, T, len(self.parts), M) > mask_p
        mask = mask.to(im_q.device)
        for i in range(len(self.parts)):
            im_q[:, :, :, self.parts[i]] = im_q[:, :, :, self.parts[i]] * mask[:, :, :, i:(i + 1)]
        return im_q

    def temporal_masking(self, im_q, mask_p):
        N, C, T, V, M = im_q.size()
        mask = torch.ones(N, 1, T, V, M).float()
        pp = np.random.random()
        if pp<mask_p:
            T_segment=3
            pint = np.random.randint(0,T_segment)
            interval = int(T/T_segment)
            # print(interval*pint,interval*(pint+1))
            mask[:,:,interval*pint:interval*(pint+1),:,:]=0.0
        mask = mask.to(im_q.device)
        im_q = im_q * mask 
        return im_q

    def regression(self, im_q, dec):
        # comparse mse in shape (N * M, T1, V * C)
        N, C, T, V, M = im_q.size()
        T1 = dec.size(1)
        ori = F.adaptive_avg_pool3d(im_q, (T1, V, M))
        ori = ori.permute(0, 4, 2, 3, 1).contiguous()
        ori = ori.view(N * M, T1, V * C)
        dec_err = F.mse_loss(dec, ori)
        return dec_err

    def make_codt_mse_loss(self, im_q, dec):
        return self.regression(im_q, dec)



























class Model_thin_G(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin_G, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        # self.fc = nn.Linear(256, num_class)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        return feature


class Predictor(nn.Module):
    def __init__(self, num_class=60, inc=256, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out



# from torch.autograd import Function
class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x,lam)










class Model_thin_ae(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
        seq_len=64):
        super(Model_thin_ae, self).__init__()

        self.encoder = Model_thin_encoder(num_class, num_point, num_person, graph, graph_args, in_channels)
        self.decoder = Model_thin_decoder(num_class, num_point, num_person, graph, graph_args, in_channels,
            seq_len)


    def forward(self, x):
        '''
        return z (latent feature), and reconstructed result
        '''
        
        # print("x_input.shape = ",x.shape)
        output_logits, feature, z = self.encoder(x, return_feat=True)
        # print("z.shape = ", z.shape)

        reconstruction = self.decoder(z)
        # print("x_recon.shape = ", reconstruction.shape)

        return reconstruction, z, output_logits


class Model_thin_encoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin_encoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        '''
        iccv21 motion consistency decoder uses deconv_tcn_gcn_unit.
        '''

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # print("x1=",x.shape)
        x = self.l1(x)
        # print("x1=",x.shape)
        x = self.l2(x)
        # print("x1=",x.shape)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # print("x1=",x.shape)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # print("x1=",x.shape)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        t_new = x.size(2)
        
        feature_with_shape = x.view(N,M,-1,t_new,V)
        # (n,c,t,v,m)
        feature_with_shape = feature_with_shape.permute(0,2,3,4,1)

        # N,M,C,T*V
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature, feature_with_shape
        else:
            return self.fc(x)



class Model_thin_decoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
        seq_len=None):
        super(Model_thin_decoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.T=seq_len

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(256, 128, A, residual=False)
        self.l2 = TCN_GCN_unit(128, 64, A, stride=2)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 64, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(64, 64, A, stride=1)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, self.T*3)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        # bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        
        feature_with_shape = x.view(N,M,-1,T,V)
        #(n,m,t,v,c)
        feature_with_shape = feature_with_shape.permute(0,1,4,2,3)
        n,m,v,_,_=feature_with_shape.shape
        feature_with_shape = feature_with_shape.reshape(n,m,v,-1)
        # print("decoder.output=",feature_with_shape.shape)
        x = self.fc(feature_with_shape)
        x = x.reshape(n,m,v,self.T,3)



        x = x.permute(0,4,3,2,1)

        return x 




























class Model_thin_TSN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model_thin_TSN, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        # print("batch inchannel: ", num_person , in_channels , num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        # self.l3 = TCN_GCN_unit(64, 64, A)
        # self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, A)
        # self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, A)
        # self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward_old(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)

    def forward_featuremap(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        feat = x 
        nm2,c2,t2,v2=feat.shape
        feat = feat.reshape(N,M,c2,t2,v2)
        # (n,c,t)
        feat = feat.mean(4).mean(1)
        # (n,t,c)
        feat = feat.permute(0,2,1).contiguous()
        featuremap = feat


        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        if return_feat:
            return self.fc(x), feature, featuremap
        else:
            return self.fc(x)


    def get_IN(self, x):
        assert len(x.shape)==4
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        eps=1e-6
        sig = (var + eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        stats=torch.cat([mu,sig],1)
        stats=stats.squeeze()
        return stats

    def forward_all_feature_stats(self, x):

        # stats1=self.get_IN(x)
        # stats2=self.get_IN(x)
        # stats3=self.get_IN(x)
        # stats4=self.get_IN(x)
        # return x, [stats1, stats2, stats3, stats4]

        N, C, T, V, M = x.size()
        assert M==1
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        stats1=self.get_IN(x)
        x = self.l2(x)
        stats2=self.get_IN(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        stats3=self.get_IN(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        stats4=self.get_IN(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.fc(x)

        return x, [stats1, stats2, stats3, stats4]
        

    def forward_each_segment(self, x):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature=x 

        # if return_feat:
        #     return self.fc(x), feature
        # else:
        #     return self.fc(x)
        return feature 

    def forward(self, x, return_feat=False):
        N, C, T, V, M = x.size()
        # print(x.size(), file=sys.stderr)

        feat_1 = self.forward_each_segment(x[:,:,:T//3,:,:])
        feat_2 = self.forward_each_segment(x[:,:,T//3:T//3*2,:,:])
        feat_3 = self.forward_each_segment(x[:,:,T//3*2:,:,:])

        feature = (feat_1 + feat_2 + feat_3) / 3
        x = feature 

        if return_feat:
            return self.fc(x), feature
        else:
            return self.fc(x)
