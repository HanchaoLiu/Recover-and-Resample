import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

# from net.utils.tgcn import ConvTemporalGraphical
# from net.utils.graph import Graph

def resize_torch_interp_batch(data, target_frame):

    window = target_frame

    n,c,t,v,m = data.shape 
    data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
    data = data[:,:,:,None]
    data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
    data = data.reshape(n,c,v,m,window)
    data = data.permute(0,1,4,2,3).contiguous()
    return data 





class AE(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, num_class, graph_args,
                 edge_importance_weighting, seqlen=8, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.seqlen = seqlen

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        print(f"temporal_kernel_size={temporal_kernel_size}!")
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels*16, kernel_size, 2, **kwargs),
        ))
        # self.fc = nn.Linear(hidden_channels*4, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True), 
            nn.Linear(512, 3*self.seqlen*25*2)
        )
        

    def encode(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            # print(x.shape)

        # (n*m,c,t,v)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        # x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        # (bs, 64)
        N,C,T,V,M = x.shape 
        latent_feature = self.encode(x)
        # print("latent_feature = ", latent_feature.shape)
        x_recon = self.decoder(latent_feature)
        x_recon = x_recon.reshape((N,3,self.seqlen,25,2))
        return x_recon

    def do_aug_random_place(self, x, squeeze_size=0.5, copy_raw=False):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    def do_random_place_shift_only(self, x, squeeze_size=0.5, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        if copy_raw:
            x_recon = mask * x #+ (1-mask) * x_recon

        return x_recon


    def do_aug_full_or_last_half_place(self, x, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1-squeeze_size)
            # ed = st + squeeze_size

            tag = np.random.choice([0,1])
            if tag==0:
                st=0.0
                ed=1.0
            elif tag==1:
                st=0.5
                ed=1.0 
            else:
                raise ValueError()

            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    
    def do_aug_random_place_first_frame_condition(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)
            st = 1 
            ed = 32
            valid_length = T - ed
            # first_frame interval
            x_squeeze[i:i+1,:,0:st,:,:] = first_frame[i:i+1,:,:,:,:]
            # original frame part 
            x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 0.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        # if copy_raw:
        #     x_recon = mask * x + (1-mask) * x_recon

        x_recon = x_squeeze

        return x_recon



    def do_aug_random_place_first_frame_condition2(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)

            ed = np.random.beta(0.1, 0.1)*0.5
            ed = int(ed*T)
            # print(ed)

            if ed==0:
                x_squeeze[i:i+1,:,:,:,:] = x[i:i+1,:,:,:,:]
            else:
                valid_length = T - ed
                # first_frame interval
                two_frame = torch.cat([first_frame[i:i+1,:,0:1,:,:], x[i:i+1,:,0:1,:,:]],2)
                x_squeeze[i:i+1,:,0:ed,:,:] = resize_torch_interp_batch(two_frame, ed)
                # original frame part 
                x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)

            

            # mask[i:i+1,:, st:ed,:,:] = 0.0

        x_recon = x_squeeze

        return x_recon




    def interpolate_w(self, x, y, w=0.5):
        N,C,T,V,M = x.shape 
        assert x.shape==y.shape 

        feat_x = self.encode(x)
        feat_y = self.encode(y)

        feat = feat_x * w + feat_y * (1-w)
        x_recon = self.decoder(feat)
        x_recon = x_recon.reshape((N,3,self.seqlen,25,2))
        return x_recon




    def forward_my(self, x, return_feat=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        feat = x 

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if return_feat:
            return x, feat
        else: 
            return x











class AE_dec(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, num_class, graph_args,
                 edge_importance_weighting, seqlen=8, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.seqlen = seqlen

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        print(f"temporal_kernel_size={temporal_kernel_size}!")
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels*16, kernel_size, 2, **kwargs),
        ))
        # self.fc = nn.Linear(hidden_channels*4, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), 
            nn.Linear(256, 3*25)
        )

        # self.adj_frame_encoder = nn.Sequential(
        #     nn.Linear(25*3*2, 256), nn.ReLU(True), 
        #     nn.Linear(256, 128)
        # )

        

    def encode(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            # print(x.shape)

        # (n*m,c,t,v)

        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(N, M, -1).mean(dim=1)

        # prediction
        # x = self.fc(x)
        # x = x.view(x.size(0), -1)

        # return x

        # (n*m,c,t,v)
        x = x.mean(dim=3)
        NM, C2, T2 = x.shape
        # x = x.view(NM, C2, T2)
        # (NM, T2, C2)
        x = x.permute(0,2,1).contiguous()
        return x 

    def forward(self, x):
        # (bs, 64)
        # frame0 = x[:,:,0:1,:,:]
        # frame1 = x[:,:,32:33,:,:]
        # x[:,:,32:,:,:] = frame1

        N,C,T,V,M = x.shape 
        # (NM,T2, M*C2)
        latent_feature = self.encode(x)
        # (NM,T2, 3*25)
        NM, T2, C2 = latent_feature.shape 
        x_recon = self.decoder(latent_feature)
        x_recon = x_recon.reshape((N*M,T2,25,3))
        x_recon = x_recon.reshape(N,M,T2,25,3)
        x_recon = x_recon.permute(0,4,2,3,1).contiguous()
        x_recon = resize_torch_interp_batch(x_recon, T)
        assert x_recon.shape==x.shape 
        return x_recon

    def forward_eval_2p(self, x):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            pos_x2     = x2[:,:,:,1:2,:]
            frame0_pos = pos_x2[:,:,0:1,:,:]
            frame1_pos = pos_x2[:,:,32:33,:,:]

            use_linear_interp=True 
            if use_linear_interp:
                steps=np.linspace(0,1,num=31,endpoint=False)
                frame_pos_steps = [(1-s)*frame0_pos+s*frame1_pos for s in steps]
                frame_pos_steps = torch.cat(frame_pos_steps, 2)
            else:
                frame_pos_steps = frame0_pos

            recon2     = x2-pos_x2 
            recon2 = self.forward(recon2)
            recon2 = recon2 + pos_x2
            recon2[:,:,1:32,:,:] = recon2[:,:,1:32,:,:] + frame_pos_steps
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 


    def forward_infill_t0(self, x, t0):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            pos_x2     = x2[:,:,:,1:2,:]
            frame0_pos = pos_x2[:,:,0:1,:,:]
            frame1_pos = pos_x2[:,:,t0:t0+1,:,:]

            use_linear_interp=True 
            if use_linear_interp and (t0>=2):
                # steps=np.linspace(0,1,num=t0-1,endpoint=False)
                steps=np.linspace(0,1,num=t0-1,endpoint=False)
                frame_pos_steps = [(1-s)*frame0_pos+s*frame1_pos for s in steps]
                frame_pos_steps = torch.cat(frame_pos_steps, 2)
            else:
                frame_pos_steps = frame0_pos

            recon2     = x2-pos_x2 
            recon2 = self.forward(recon2)
            recon2 = recon2 + pos_x2
            if t0>=2:
                recon2[:,:,1:t0,:,:] = recon2[:,:,1:t0,:,:] + frame_pos_steps
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 
    


    def forward_extrap(self, x, t0):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            recon2 = self.forward(x2)
            
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 
    


    def do_aug_random_place(self, x, squeeze_size=0.5, copy_raw=False):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    def do_random_place_shift_only(self, x, squeeze_size=0.5, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        if copy_raw:
            x_recon = mask * x #+ (1-mask) * x_recon

        return x_recon


    def do_aug_full_or_last_half_place(self, x, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1-squeeze_size)
            # ed = st + squeeze_size

            tag = np.random.choice([0,1])
            if tag==0:
                st=0.0
                ed=1.0
            elif tag==1:
                st=0.5
                ed=1.0 
            else:
                raise ValueError()

            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    
    def do_aug_random_place_first_frame_condition(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)
            st = 1 
            ed = 32
            valid_length = T - ed
            # first_frame interval
            x_squeeze[i:i+1,:,0:st,:,:] = first_frame[i:i+1,:,:,:,:]
            # original frame part 
            x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 0.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        # if copy_raw:
        #     x_recon = mask * x + (1-mask) * x_recon

        x_recon = x_squeeze

        return x_recon



    def do_aug_random_place_first_frame_condition2(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)

            ed = np.random.beta(0.1, 0.1)*0.5
            ed = int(ed*T)
            # print(ed)

            if ed==0:
                x_squeeze[i:i+1,:,:,:,:] = x[i:i+1,:,:,:,:]
            else:
                valid_length = T - ed
                # first_frame interval
                two_frame = torch.cat([first_frame[i:i+1,:,0:1,:,:], x[i:i+1,:,0:1,:,:]],2)
                x_squeeze[i:i+1,:,0:ed,:,:] = resize_torch_interp_batch(two_frame, ed)
                # original frame part 
                x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)

            

            # mask[i:i+1,:, st:ed,:,:] = 0.0

        x_recon = x_squeeze

        return x_recon




    def interpolate_w(self, x, y, w=0.5):
        N,C,T,V,M = x.shape 
        assert x.shape==y.shape 

        feat_x = self.encode(x)
        feat_y = self.encode(y)

        feat = feat_x * w + feat_y * (1-w)
        x_recon = self.decoder(feat)
        x_recon = x_recon.reshape((N,3,self.seqlen,25,2))
        return x_recon




    def forward_my(self, x, return_feat=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        feat = x 

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if return_feat:
            return x, feat
        else: 
            return x





class AE_dec_kn12(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, num_class, graph_args,
                 edge_importance_weighting, seqlen=8, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.seqlen = seqlen

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        print(f"temporal_kernel_size={temporal_kernel_size}!")
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels*16, kernel_size, 2, **kwargs),
        ))
        # self.fc = nn.Linear(hidden_channels*4, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), 
            # nn.Linear(256, 3*25)
            nn.Linear(256, 2*18)
        )

        # self.adj_frame_encoder = nn.Sequential(
        #     nn.Linear(25*3*2, 256), nn.ReLU(True), 
        #     nn.Linear(256, 128)
        # )

        

    def encode(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            # print(x.shape)

        # (n*m,c,t,v)

        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(N, M, -1).mean(dim=1)

        # prediction
        # x = self.fc(x)
        # x = x.view(x.size(0), -1)

        # return x

        # (n*m,c,t,v)
        x = x.mean(dim=3)
        NM, C2, T2 = x.shape
        # x = x.view(NM, C2, T2)
        # (NM, T2, C2)
        x = x.permute(0,2,1).contiguous()
        return x 

    def forward(self, x):
        # (bs, 64)
        # frame0 = x[:,:,0:1,:,:]
        # frame1 = x[:,:,32:33,:,:]
        # x[:,:,32:,:,:] = frame1

        # print("hello!")

        N,C,T,V,M = x.shape 
        # (NM,T2, M*C2)
        latent_feature = self.encode(x)
        # (NM,T2, 3*25)
        NM, T2, C2 = latent_feature.shape 
        x_recon = self.decoder(latent_feature)
        # x_recon = x_recon.reshape((N*M,T2,25,3))
        # x_recon = x_recon.reshape(N,M,T2,25,3)
        x_recon = x_recon.reshape((N*M,T2,18,2))
        x_recon = x_recon.reshape(N,M,T2,18,2)
        x_recon = x_recon.permute(0,4,2,3,1).contiguous()
        x_recon = resize_torch_interp_batch(x_recon, T)
        assert x_recon.shape==x.shape 
        return x_recon

    def forward_eval_2p(self, x):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            pos_x2     = x2[:,:,:,1:2,:]
            frame0_pos = pos_x2[:,:,0:1,:,:]
            frame1_pos = pos_x2[:,:,32:33,:,:]

            use_linear_interp=True 
            if use_linear_interp:
                steps=np.linspace(0,1,num=31,endpoint=False)
                frame_pos_steps = [(1-s)*frame0_pos+s*frame1_pos for s in steps]
                frame_pos_steps = torch.cat(frame_pos_steps, 2)
            else:
                frame_pos_steps = frame0_pos

            recon2     = x2-pos_x2 
            recon2 = self.forward(recon2)
            recon2 = recon2 + pos_x2
            recon2[:,:,1:32,:,:] = recon2[:,:,1:32,:,:] + frame_pos_steps
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 


    def forward_infill_t0(self, x, t0):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            pos_x2     = x2[:,:,:,1:2,:]
            frame0_pos = pos_x2[:,:,0:1,:,:]
            frame1_pos = pos_x2[:,:,t0:t0+1,:,:]

            use_linear_interp=True 
            if use_linear_interp and (t0>=2):
                # steps=np.linspace(0,1,num=t0-1,endpoint=False)
                steps=np.linspace(0,1,num=t0-1,endpoint=False)
                frame_pos_steps = [(1-s)*frame0_pos+s*frame1_pos for s in steps]
                frame_pos_steps = torch.cat(frame_pos_steps, 2)
            else:
                frame_pos_steps = frame0_pos

            recon2     = x2-pos_x2 
            recon2 = self.forward(recon2)
            recon2 = recon2 + pos_x2
            if t0>=2:
                recon2[:,:,1:t0,:,:] = recon2[:,:,1:t0,:,:] + frame_pos_steps
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 
    


    def forward_extrap(self, x, t0):
        x1 = x[:,:,:,:,0:1]
        x2 = x[:,:,:,:,1:2]
        recon1 = self.forward(x1)

        # ref_joint=1
        # pos_x2 = x2[:,:,:,ref_joint:ref_joint+1,:]
        # x2_centered = x2 - pos_x2
        # # recon2 = self.forward(x2_centered)
        # recon2 = x2_centered
        # recon2 = recon2 + pos_x2
        # recon2[:,:,1:32,:,:] = (x2[:,:,0:1,:,:]+x2[:,:,32:33,:,:])/2

        # remove trajectory for x2 first
        # apply infilling 
        # restore trajectory for x2 
        # since 1:32 are masks originally, we have to add trajectory of first frame.

        if x2.sum()!=0:
            recon2 = self.forward(x2)
            
        else:
            recon2 = x2 

        res = torch.cat([recon1, recon2], 4)
        assert res.shape==x.shape 
        return res 
    


    def do_aug_random_place(self, x, squeeze_size=0.5, copy_raw=False):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    def do_random_place_shift_only(self, x, squeeze_size=0.5, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            st = np.random.uniform(0,1-squeeze_size)
            ed = st + squeeze_size
            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        if copy_raw:
            x_recon = mask * x #+ (1-mask) * x_recon

        return x_recon


    def do_aug_full_or_last_half_place(self, x, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.zeros_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1-squeeze_size)
            # ed = st + squeeze_size

            tag = np.random.choice([0,1])
            if tag==0:
                st=0.0
                ed=1.0
            elif tag==1:
                st=0.5
                ed=1.0 
            else:
                raise ValueError()

            st = int(st*T)
            ed = int(ed*T)
            valid_length = ed - st 
            x_squeeze[i:i+1,:,st:ed,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 1.0

        is_training = self.training 
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x_squeeze)
        if is_training:
            self.train()

        if copy_raw:
            x_recon = mask * x + (1-mask) * x_recon

        return x_recon

    
    def do_aug_random_place_first_frame_condition(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)
            st = 1 
            ed = 32
            valid_length = T - ed
            # first_frame interval
            x_squeeze[i:i+1,:,0:st,:,:] = first_frame[i:i+1,:,:,:,:]
            # original frame part 
            x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)
            mask[i:i+1,:, st:ed,:,:] = 0.0

        # is_training = self.training 
        # self.eval()
        # with torch.no_grad():
        #     x_recon = self.forward(x_squeeze)
        # if is_training:
        #     self.train()

        # if copy_raw:
        #     x_recon = mask * x + (1-mask) * x_recon

        x_recon = x_squeeze

        return x_recon



    def do_aug_random_place_first_frame_condition2(self, x, first_frame, copy_raw=True):
        # do for every sample 
        N,C,T,V,M = x.shape 
        # generate squeezed x
        x_squeeze = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for i in range(N):
            # st = np.random.uniform(0,1/16)
            # ed = np.random.uniform(0.25, 0.75)
            # st = int(st*T)
            # ed = int(ed*T)

            ed = np.random.beta(0.1, 0.1)*0.5
            ed = int(ed*T)
            # print(ed)

            if ed==0:
                x_squeeze[i:i+1,:,:,:,:] = x[i:i+1,:,:,:,:]
            else:
                valid_length = T - ed
                # first_frame interval
                two_frame = torch.cat([first_frame[i:i+1,:,0:1,:,:], x[i:i+1,:,0:1,:,:]],2)
                x_squeeze[i:i+1,:,0:ed,:,:] = resize_torch_interp_batch(two_frame, ed)
                # original frame part 
                x_squeeze[i:i+1,:,ed:T,:,:] = resize_torch_interp_batch(x[i:i+1], valid_length)

            

            # mask[i:i+1,:, st:ed,:,:] = 0.0

        x_recon = x_squeeze

        return x_recon




    def interpolate_w(self, x, y, w=0.5):
        N,C,T,V,M = x.shape 
        assert x.shape==y.shape 

        feat_x = self.encode(x)
        feat_y = self.encode(y)

        feat = feat_x * w + feat_y * (1-w)
        x_recon = self.decoder(feat)
        x_recon = x_recon.reshape((N,3,self.seqlen,25,2))
        return x_recon




    def forward_my(self, x, return_feat=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        feat = x 

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if return_feat:
            return x, feat
        else: 
            return x








class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 1
        print(f"temporal_kernel_size={temporal_kernel_size}!")
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

    def forward_my(self, x, return_feat=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        feat = x 

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if return_feat:
            return x, feat
        else: 
            return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


















class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


# The based unit of graph convolutional networks.

# import torch
# import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A



def main():

    graph_args=dict(layout='ntu-rgb+d',strategy='spatial')
    m = AE(in_channels=3, hidden_channels=16, num_class=60, graph_args=graph_args,
                 edge_importance_weighting=True)
    x = torch.zeros((8,3,8,25,2))
    y = m(x)
    print("input.shape=", x.shape)
    print("output.shape=", y.shape)


if __name__ == "__main__":
    main()