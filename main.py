from utils.utils import make_parse
from utils.core import test
from torch.utils.data import DataLoader
from datasets.datasets import h5file_Dataset
from models.models import MultipleMILTransformer as MMILT
import torch
import numpy as np

def main(args):
    torch.manual_seed(2023)
    model = MMILT(args).cuda()
    data_csv_dir = args.csv
    h5file_dir = args.h5
    
    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    loader = test_dataloader
    
    acc = []
    auc = []
    for i in range(args.num_test):
        model.load_state_dict(torch.load(args.test))
        test_acc,test_auc = test(args,model,loader)
        acc.append(test_acc)
        auc.append(test_auc.cpu())
    print('Average acc and auc for {} times test is {} and {}'.format(args.num_test,np.mean(acc),np.mean(auc)))
        
       
args = make_parse()
main(args)