from __future__ import division
import os
import argparse

import pandas as pd
import glob
import nibabel as nib
from tqdm import tqdm
import numpy as np
import copy
import yaml
from types import SimpleNamespace  
import trainer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from loader import *
from networks.attention_swin_unet import SwinAttentionUnet as ViT_seg
from configs import swin_attention_unet as config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./Synapse/', help='root dir for data')
parser.add_argument('--eval_interval', type=int, default=5, help='eval interval')
parser.add_argument('--volume_path', type=str,
                    default='./Synapse/', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--saved_model', type=str,
                    default='./weights/weights_isic17.model' , help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--mode', help='Select our contribution',
                    choices=['swin','cross_contextual_attention'], default='swin')
parser.add_argument('--skip_num', help='Select our contribution',
                    choices=['0', '1', '2','3'], default='3'),
parser.add_argument('--operationaddatten', help='Select our contribution',
                    choices=['+', 'mul'], default='+')
parser.add_argument('--attention', help='0 or 1',
                    choices=['0',"1"], default="0")

args = parser.parse_args(args=[])
config =  config.get_swin_unet_attention_configs().to_dict()
config.update(vars(args))
configs = SimpleNamespace(**config)
#skin config
config         = yaml.load(open('./configs/config_skin.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path     = config['path_to_data']  
train_dataset = isic_loader(path_Data = data_path, train = True)
train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True)
val_dataset   = isic_loader(path_Data = data_path, train = False)
val_loader    = DataLoader(val_dataset, batch_size = int(config['batch_size_va']), shuffle= False)
test_dataset  = isic_loader(path_Data = data_path, train = False, Test = True)
test_loader   = DataLoader(test_dataset, batch_size = 1, shuffle= True)
#build model
Net   = ViT_seg(configs,num_classes=args.num_classes).cuda()
Net   = Net.to(device)
if int(config['pretrained']):
    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']

optimizer = optim.Adam(Net.parameters(), lr= float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = config['patience'])
criteria  = torch.nn.BCELoss()
trainer(config,Net,train_loader,test_loader,optimizer,criteria,configs)
        
        
        
        

