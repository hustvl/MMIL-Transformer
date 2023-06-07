import torchmetrics
import torch.nn as nn
import torch
from .utils import *
import numpy as np
from sklearn.cluster import KMeans

torch.manual_seed(2023)

def test(args,model,dataloader):
    np.random.seed(args.seed)
    print('-------testing-------')
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = dataloader
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
        for idx, (coords ,data, label) in enumerate(test_loader):
            coords ,data, label =coords.to(device), data.to(device), label.to(device).long()
            results_dict = model(data,coords,mask_ratio=0)
            logits,Y_prob,Y_hat = results_dict['logits'],results_dict['Y_prob'],results_dict['Y_hat']
            if idx == 0:
                Y_prob_list = Y_prob
                label_list = label
            else:
                Y_prob_list = torch.cat((Y_prob_list,Y_prob), dim=0)
                label_list = torch.cat((label_list,label), dim=0)
            loss = loss_fn(logits,label)
            test_loss += loss
            error = calculate_error(Y_hat,label)
            test_error += error
        test_auroc = torchmetrics.AUROC(num_classes=2)
        test_auc = test_auroc(Y_prob_list, label_list)

    t_hit_num = len(test_loader) - test_error
    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('test_loss: {:.4f}, test_error: {:.4f}'.format(test_loss, test_error))
    print('test_correct:',int(t_hit_num),'/',len(test_loader))
    print('test_auc: {}'.format(test_auc))
    print('-----------------------')
    
    return 1-test_error, test_auc
    
class grouping:

    def __init__(self,groups_num,max_size=1e10):
        self.groups_num = groups_num
        self.max_size = int(max_size) # Max lenth 4300 for 24G RTX3090
        
    
    def indicer(self, labels):
        indices = []
        groups_num = len(set(labels))
        for i in range(groups_num):
            temp = np.argwhere(labels==i).squeeze()
            indices.append(temp)
        return indices
    
    def make_subbags(self, idx, features):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            if member_size > self.max_size:
                index[i] = np.random.choice(index[i],size=self.max_size,replace=False)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0)
            features_group.append(temp)
            
        return features_group
        
    def coords_nomlize(self, coords):
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
    
    
    def coords_grouping(self,coords,features,c_norm=False):
        features = features.squeeze()
        coords = coords.squeeze()
        if c_norm:
            coords = self.coords_nomlize(coords.float())
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(coords.cpu().numpy())
        indices = self.indicer(k.labels_)
        
        return indices
    
    def embedding_grouping(self,features):
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(features.cpu().detach().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)

        return features_group
    
    def random_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = split_array(np.array(range(int(N))),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group
        
    def seqential_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = np.array_split(range(N),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group