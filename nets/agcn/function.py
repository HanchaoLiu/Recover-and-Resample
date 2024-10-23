import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())



class Encoder(nn.Module):
    def __init__(self, n_features, seq_len=64):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features

        
        self.hidden_dim_1 = 64
        self.hidden_dim_2 = 16

        self.n_layers = 1

        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim_1,
          num_layers=self.n_layers,
          batch_first=True  # True = (batch_size, seq_len, n_features)
                            # False = (seq_len, batch_size, n_features) 
                            #default = false
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim_1,
          hidden_size=self.hidden_dim_2,
          num_layers=self.n_layers,
          batch_first=True
        )

    def forward(self, x):

        batch_size = x.shape[0]
        #(4,1)
        # x = x.reshape((batch_size, self.seq_len, self.n_features)) 
        # (batch, seq, feature)   #(1,4,1) 
        x, (_, _) = self.rnn1(x) #(1,4,256) 
        x, (hidden_n, _) = self.rnn2(x)
        # x shape (1,4,128)
        # hidden_n (1,1,128)
        # return hidden_n.reshape((self.n_features, self.embedding_dim)) #(1,128)
        return x 


class Decoder(nn.Module):
    def __init__(self, n_features, seq_len=64):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.n_layers=1

        self.rnn1 = nn.LSTM(
            input_size=16,
            hidden_size=64,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=64,
            hidden_size=75,
            num_layers=self.n_layers,
            batch_first=True
        )
        

    def forward(self, x):
        # x shape(1,128)
        # x = x.repeat(self.seq_len, args.batch_size) # (4, 128)
        x, (_, _) = self.rnn1(x) #(1,4,256) 
        x, (hidden_n, _) = self.rnn2(x)
        return x 
       


class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, seq_len=64):
        super(RecurrentAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.encoder = Encoder(n_features, seq_len)
        self.decoder = Decoder(n_features, seq_len)

    def forward(self, x):
        x = self.nctvm_to_nSeqlenFeat(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.nSeqlenFeat_to_nctvm(x)
        return x, None, None

    @staticmethod
    def nctvm_to_nSeqlenFeat(x):
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(x.shape[0],x.shape[1],-1)
        return x 
    
    @staticmethod
    def nSeqlenFeat_to_nctvm(x):
        n,seqlen,feat_dim=x.shape 
        # (n,t,v,c,m)
        x = x.view(n,seqlen,25,3,1)
        x = x.permute(0,3,1,2,4).contiguous()
        return x 


if __name__ == "__main__":
    m = RecurrentAutoencoder(n_features=75)
    m.cuda()
    x = torch.randn(size=(2,3,64,25,1)).cuda()
    # x = torch.randn(size=(2,64,75))
    y = m(x)
    print("x=",x.shape)
    print("y=",y.shape)