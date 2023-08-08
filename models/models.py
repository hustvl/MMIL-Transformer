from utils.core import *
from utils.utils import *
import torch
import random
import torch.nn as nn
from einops import rearrange
from nystrom_attention import NystromAttention
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #x = x.squeeze(dim=0)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class AttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1,attn_mode='normal'):
        super(AttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.mode = attn_mode
        self.attn = Attention(self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
    def forward(self,x):
        return x + self.attn(x)

class NyAttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super(NyAttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
    def forward(self,x):
        return x + self.attn(x)

class GroupsAttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1,attn_mode='normal'):
        super(GroupsAttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        if attn_mode == 'nystrom':
            self.AttenLayer = NyAttenLayer(dim =self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
        else:
            self.AttenLayer = AttenLayer(dim =self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)

    def forward(self,x_groups,mask_ratio=0):
        group_after_attn = []
        r = int(len(x_groups) * (1-mask_ratio))
        x_groups_masked = random.sample(x_groups, k=r)
        for x in x_groups_masked:
            x = x.squeeze(dim=0)
            temp = self.AttenLayer(x).unsqueeze(dim=0)
            group_after_attn.append(temp)
        return group_after_attn
    
class GroupsMSGAttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.AttenLayer = AttenLayer(dim =self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
    def forward(self,data):
        msg_cls, x_groups, msg_tokens_num = data
        groups_num = len(x_groups)
        msges = torch.zeros(size=(1,1,groups_num*msg_tokens_num,self.dim)).to(msg_cls.device)
        for i in range(groups_num):
            msges[:,:,i*msg_tokens_num:(i+1)*msg_tokens_num,:] = x_groups[i][:,:,0:msg_tokens_num]
        msges = torch.cat((msg_cls,msges),dim=2).squeeze(dim=0)
        msges = self.AttenLayer(msges).unsqueeze(dim=0)
        msg_cls = msges[:,:,0].unsqueeze(dim=0)
        msges = msges[:,:,1:]
        for i in range(groups_num):
            x_groups[i] = torch.cat((msges[:,:,i*msg_tokens_num:(i+1)*msg_tokens_num],x_groups[i][:,:,msg_tokens_num:]),dim=2)
        data = msg_cls, x_groups, msg_tokens_num
        return data
    
class BasicLayer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.GroupsAttenLayer = GroupsAttenLayer(dim=dim)
        self.GroupsMSGAttenLayer = GroupsMSGAttenLayer(dim=dim)
    def forward(self,data,mask_ratio):
        msg_cls, x_groups, msg_tokens_num = data
        x_groups = self.GroupsAttenLayer(x_groups,mask_ratio)
        data = (msg_cls, x_groups, msg_tokens_num)
        data = self.GroupsMSGAttenLayer(data)
        return data


class MultipleMILTransformer(nn.Module):
    def __init__(self,args):
        super(MultipleMILTransformer, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.in_chans, self.args.embed_dim)
        self.fc2 = nn.Linear(self.args.embed_dim, self.args.n_classes)
        self.msg_tokens_num = self.args.num_msg
        self.msgcls_token = nn.Parameter(torch.randn(1,1,1,self.args.embed_dim))
        #---> make sub-bags
        print('try to group seq to ',self.args.num_subbags)
        self.grouping = grouping(self.args.num_subbags,max_size=4300)
        if self.args.mode == 'random':
            self.grouping_features = self.grouping.random_grouping
        elif self.args.mode == 'coords':
            self.grouping_features = self.grouping.coords_grouping
        elif self.args.mode == 'seq':
            self.grouping_features = self.grouping.seqential_grouping
        elif self.args.mode == 'embed':
            self.grouping_features = self.grouping.embedding_grouping
        elif self.args.mode == 'idx':
            self.grouping_features = self.grouping.idx_grouping
        self.msg_tokens = nn.Parameter(torch.zeros(1, 1, 1, self.args.embed_dim))
        self.cat_msg2cluster_group = cat_msg2cluster_group
        if self.args.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 1, self.args.embed_dim))
        
        #--->build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.args.num_layers):
            layer = BasicLayer(dim=self.args.embed_dim)
            self.layers.append(layer)

    def head(self,x):
        logits = self.fc2(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


    def forward(self,x,coords=False,mask_ratio=0):
        #---> init
        if self.args.type == 'camelyon16':
            x = self.fc1(x)
        else:
            x = x.float()
        if self.args.ape:
            x = x + self.absolute_pos_embed.expand(1,x.shape[1],self.args.embed_dim)
        if self.args.mode == 'coords' or self.args.mode == 'idx':
            x_groups = self.grouping_features(coords,x) 
        else:
            x_groups = self.grouping_features(x)
        msg_tokens = self.msg_tokens.expand(1,1,self.msg_tokens_num,-1)
        msg_cls = self.msgcls_token
        x_groups = self.cat_msg2cluster_group(x_groups,msg_tokens)
        data = (msg_cls, x_groups, self.msg_tokens_num)
        #---> feature forward
        for i in range(len(self.layers)):
            if i == 0:
                mr = mask_ratio
                data = self.layers[i](data,mr)
            else:
                mr = 0
                data = self.layers[i](data,mr)
        #---> head
        msg_cls, _, _ = data
        msg_cls = msg_cls.view(1,self.args.embed_dim)
        results_dict = self.head(msg_cls)
        #print(results_dict)

        return results_dict
