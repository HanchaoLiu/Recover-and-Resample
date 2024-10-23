# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils import utils
import torchvision
import os



class HCN_jigsaw(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 n_jigsaw = 2,
                 ):
        super(HCN_jigsaw, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.n_jigsaw = n_jigsaw
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16),256*2), # 4*4 for window=64; 8*8 for window=128
            nn.PReLU(),
            nn.Dropout2d(p=0.5))

        self.fc7_rl= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16),256*2), # 4*4 for window=64; 8*8 for window=128
            nn.PReLU(),
            nn.Dropout2d(p=0.5))
        
        self.fc8 = nn.Linear(256*2,num_class)

        num_class_rl=self.n_jigsaw
        self.fc8_rl = nn.Linear(256*2,num_class_rl)

        # initial weight
        initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,return_feat=False):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0],logits[1])
        out_conv6 = out
        out = out.view(out.size(0), -1)

        out = self.fc7(out)
        feature = out  #tsne
        out_class = self.fc8(out)
        out_rl = self.fc8_rl(out)
        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        if return_feat:
            return out_class,out_rl,feature
        else:
            return out_class,out_rl


class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 ):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16),256*2), # 4*4 for window=64; 8*8 for window=128
            nn.PReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        # utils.initial_model_weight(layers = list(self.children()))
        initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,return_feat=False):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0],logits[1])
        out_conv6 = out

        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        feature = out
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        if return_feat:
            return out, feature
        else:
            return out 

        # return out,feature,out_conv6






class HCN_2(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 ):
        super(HCN_2, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.PReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16),256*2), # 4*4 for window=64; 8*8 for window=128
            nn.PReLU(),)
            # nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        # utils.initial_model_weight(layers = list(self.children()))
        initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,return_feat=False):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0],logits[1])
        out_conv6 = out

        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        feature = out
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        if return_feat:
            return out, feature
        else:
            return out 

        # return out,feature,out_conv6




# initial way
# use the next 3 functions to initial a model, eg at the third function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)


def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        bsize = input.size(0)
        tsize = input.size(1)
        nsize = input.size(2)

        #print('bsize')
        #print(bsize)

        #print('nsize')
        #print(nsize)

        #print('wsize')
        #print(self.weight.shape)

        #print('inputsize')
        #print(input.shape)

        support = torch.mm(input.view(bsize * tsize *nsize, -1), self.weight).view(bsize,tsize ,  nsize, -1)   # (bsize, nsize, outsize)
        output = torch.matmul(adj, support)
        #print('outsize')
        #print(output.shape)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def b_forward(self, inputs, adjs):
        outputs = []
        for i in range(inputs.size(0)):
            input = inputs[i]
            adj = adjs[i] if adjs.dim() == 3 else adjs
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                outputs.append(output.unsqueeze(0) + self.bias)
            else:
                return outputs.append(output.unsqueeze(0))
        return torch.cat(outputs, dim=0)

    def _forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GRL(torch.autograd.Function):

    def __init__(self, high_value=1.0, max_iter=100.0): #10000
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = max_iter

 
    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output
 

    def backward(self, gradOutput):
        # print("---grl---")
        if self.iter_num >= self.max_iter:
            self.iter_num = self.max_iter
        self.coeff = np.float(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        #print(self.coeff)
        return -(self.coeff) * gradOutput 
 

class AdversarialNetwork(nn.Module):
    def __init__(self, feature_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(feature_size, 1024)
        self.ad_layer2 = nn.Linear(1024, 512) #change here from 256
        self.ad_layer3 = nn.Linear(512, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
 

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x




def loss_fn(outputs,labels,current_epoch=None,params=None):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    if params.loss_args["type"] == 'CE':
        CE = nn.CrossEntropyLoss()(outputs, labels)
        loss_all = CE
        loss_bag = {'ls_all': loss_all, 'ls_CE': CE}
    #elif: other losses

    return loss_bag


def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop2(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop3(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop5': accuracytop5,
    # could add more metrics such as accuracy for each token type
}

if __name__ == '__main__':
    model = HCN()
    children = list(model.children())
    print(children)
