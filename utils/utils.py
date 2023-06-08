import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='TCGA',type=str)
    parser.add_argument('--mode', default='random',type=str)
    parser.add_argument('--in_chans', default=1024,type=int)
    parser.add_argument('--num_subbags', default=16,type=int)
    parser.add_argument('--embed_dim', default=512,type=int)
    parser.add_argument('--attn', default='normal',type=str)
    parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--num_layers', default=2,type=int)
    parser.add_argument('--h5', default='./h5_dir',type=str)
    parser.add_argument('--csv', default='./data.csv',type=str)
    parser.add_argument('--seed', default=2087,type=int)
    parser.add_argument('--test', default=None,type=str)
    parser.add_argument('--num_test', default=10,type=int)
    args = parser.parse_args()
    return args

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error

def coords_nomlize(coords):
    coords = coords.squeeze()
    means = torch.mean(coords,0)
    xmean,ymean = means[0],means[1]
    stds = torch.std(coords,0)
    xstd,ystd = stds[0],stds[1]
    xcoords = (coords[:,0] - xmean)/xstd
    ycoords = (coords[:,1] - ymean)/ystd
    xcoords,ycoords = xcoords.view(xcoords.shape[0],1),ycoords.view(ycoords.shape[0],1)
    coords = torch.cat((xcoords,ycoords),dim=1)
    return coords

def shuffle_msg(x):
    # (B, G, win**2+1, C)
    x = x.unsqueeze(dim=0)
    B, G, N, C = x.shape
    if G == 1:
        return x
    msges = x[:, :, 0] # (B, G, C)
    assert C % G == 0
    msges = msges.view(-1, G, G, C//G).transpose(1, 2).reshape(B, G, 1, C)
    print(msges.shape)
    x = torch.cat((msges, x[:, :, 1:]), dim=2)
    x = x.squeeze(dim=0)
    return x

def padding(h):
    H = h.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    h = torch.cat([h, h[:,:add_length,:]],dim = 1)
    return h,_H,_W

def cat_msg2cluster_group(x_groups,msg_tokens):
    x_groups_cated = []
    for x in x_groups:
        x = x.unsqueeze(dim=0)
        try:
            temp = torch.cat((msg_tokens,x),dim=2)
        except Exception as e:
            print('Error when cat msg tokens to sub-bags')
        x_groups_cated.append(temp)

    return x_groups_cated



def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False)
    split_indices = np.array_split(indices, m)  

    result = []
    for indices in split_indices:
        result.append(array[indices])

    return result

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.flag = False

    def __call__(self, epoch, val_loss, model, args, ckpt_name = ''):
        ckpt_name = './ckp/{}_checkpoint_{}_{}.pt'.format(str(args.type),str(args.seed),str(epoch))
        score = -val_loss
        self.flag = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
            self.counter = 0
        

    def save_checkpoint(self, val_loss, model, ckpt_name, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose and not args.overfit:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)
        elif self.verbose and args.overfit:
            print(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)           
        torch.save(model.state_dict(), ckpt_name)
        print(ckpt_name)
        self.val_loss_min = val_loss
        self.flag = True
