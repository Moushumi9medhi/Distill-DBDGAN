# -*- coding: utf-8 -*-
"""
*****************************************

Distill-DBDGAN:
Knowledge Distillation and Adversarial Learning Framework for Defocus Blur Detection

*****************************************
*****************************************

DEVELOPER: SANKARAGANESH JONNA, MOUSHUMI MEDHI

*****************************************
"""

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

import functools
# MODULES
from model import *
from dataloaders.DUT_loader import DUTBlur
from dataloaders.CUHK_loader import CUHKBlur
from dataloaders.SZU_BD_loader import SZU_BDBlur
from EfficientUnet import *
from dataloaders.utils import * 

# CONFIG
import argparse
arg = argparse.ArgumentParser(description='defocus blur detection')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='test_DUT.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")     
print(device) 
print(torch.cuda.is_available()) 
# MINIMIZE RANDOMNESS
np.random.seed(config.seed)
torch.manual_seed(config.seed)


if config.use_gpu:
    if len(config.gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)

test = config.test


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def test_main(args):
    if args.test_option=='DUT':
        print("DUT DATA BLUR DETECTION**************")
        data_test = DUTBlur(args)
        
    elif args.test_option=='CUHK':
        print("CUHK DATA BLUR DETECTION**************")
        data_test = CUHKBlur(args)
    else:
        print("Su-BD DATA BLUR DETECTION**************")
        data_test = SZU_BDBlur(args)
   
    
    loader_test = DataLoader(dataset=data_test,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers)

    Student = get_efficientunet_b3(out_channels=1, concat_input=True, pretrained=True).cuda()
    Student = nn.DataParallel(Student, device_ids = [0]).cuda()
    Student = Student.to(device)

  
    # LOAD MODEL
    if test:
        modeltrained = os.path.join( args.test_model_path, 'Student_model.pth')
        
        assert os.path.exists(modeltrained), \
            "file not found: {}".format(modeltrained)
        test_file = os.path.join( modeltrained)
        print(test_file)
        if os.path.isfile(test_file):
            print("=> loading checkpoint '{}'".format(test_file))
            checkpoint = torch.load(test_file)
            Student.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}' ".format(test_file))
    
    num_sample = len(loader_test) * loader_test.batch_size
        
    print('Number of samples in the test data: ', num_sample)

    Student.eval()
    
    with torch.no_grad():
        for batch_, sample_ in enumerate(loader_test):

            torch.cuda.synchronize()
            Blur, labels  = sample_
            Blur = Blur.to(device)
            labels = labels.to(device)
            Blur = Blur.cuda()
            labels = labels.cuda()
            
            Blur = Variable( Blur)
            labels = Variable( labels)
            
            g1_output = Student(Blur)
            
            if args.save_test_image:
                saveOutput(batch_,  g1_output)

            del Blur
            del g1_output
           
 

        
        
# Define the main function
def main(args):
    test_main(args)
    
          
if __name__ == "__main__":
    main(config)