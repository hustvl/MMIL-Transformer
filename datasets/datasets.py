from torch.utils.data import Dataset
import pandas as pd
import h5py, os
import numpy as np
import torch

class h5file_Dataset(Dataset):
    def __init__(self, csv_file, h5file_dir, datatype):
        self.csv_file = pd.read_csv(csv_file)
        self.h5file_dir = h5file_dir
        self.datatype = datatype
        if self.datatype == 'train':
            self.csv_index = [self.csv_file.columns.get_loc('train'),self.csv_file.columns.get_loc('train_label')]
            self.lenth = self.csv_file['train'].count()
        elif self.datatype == 'test':
            self.csv_index = [self.csv_file.columns.get_loc('test'),self.csv_file.columns.get_loc('test_label')]
            self.lenth = self.csv_file['test'].count()
        elif self.datatype == 'val':
            self.csv_index = [self.csv_file.columns.get_loc('val'),self.csv_file.columns.get_loc('val_label')]
            self.lenth = self.csv_file['val'].count()
    def __len__(self):
        return self.lenth
    
    def __getitem__(self, index):
        data_dir = os.path.join(self.h5file_dir, self.csv_file.iloc[index, self.csv_index[0]])
        data = h5py.File(data_dir+'.h5')
        features = np.array(data['features'])
        coords = np.array(data['coords'])
        label = self.csv_file.iloc[index, self.csv_index[1]]
        return coords, features, label


class h5file_Dataset_with_Cluster_index(Dataset):
    def __init__(self, csv_file, h5file_dir, clusteridx_dir,datatype):
        self.csv_file = pd.read_csv(csv_file)
        self.h5file_dir = h5file_dir
        self.datatype = datatype
        self.clusteridx_dir = clusteridx_dir
        if self.datatype == 'train':
            self.csv_index = [self.csv_file.columns.get_loc('train'),self.csv_file.columns.get_loc('train_label')]
            self.lenth = self.csv_file['train'].count()
        elif self.datatype == 'test':
            self.csv_index = [self.csv_file.columns.get_loc('test'),self.csv_file.columns.get_loc('test_label')]
            self.lenth = self.csv_file['test'].count()
        elif self.datatype == 'val':
            self.csv_index = [self.csv_file.columns.get_loc('val'),self.csv_file.columns.get_loc('val_label')]
            self.lenth = self.csv_file['val'].count()

    def __len__(self):
        return self.lenth
    
    def __getitem__(self, index):
        data_dir = os.path.join(self.h5file_dir, self.csv_file.iloc[index, self.csv_index[0]])
        data_group_idx_dir = os.path.join(self.clusteridx_dir, self.csv_file.iloc[index, self.csv_index[0]])
        data = h5py.File(data_dir+'.h5')
        data_group_idx = np.load(data_group_idx_dir+'.npy')
        features = np.array(data['features'])
        label = self.csv_file.iloc[index, self.csv_index[1]]
        return data_group_idx,features,label
    
