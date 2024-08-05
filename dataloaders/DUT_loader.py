"""
*****************************************

Distill-DBDGAN:
DUT loader
*****************************************
*****************************************

DEVELOPER: SANKARAGANESH JONNA, MOUSHUMI MEDHI

*****************************************
"""

import torch.utils.data as data
from dataloaders.paths_and_transform import *

from utils import * 




class DUTBlur(data.Dataset):
    """A data loader for the DUT dataset
    """

    def __init__(self, args):
        self.args = args
        self.test_set_path = blur_dataset( args)
        
        self.transform = blurdatatransforms 

    
    def __getitem__(self, item):
        original_img_path, shadow_mask_path = self.test_set_path[item]
        
        original_img = read_BLUR(original_img_path)
        shadow_mask = read_BLUR(shadow_mask_path)
        
        original_img = (resizeBLUR(original_img))
        shadow_mask = (resizeBLUR(shadow_mask))
        
        original_img = self.transform((original_img))
        shadow_mask = self.transform((shadow_mask))

        
        shadow_mask = shadow_mask[0:1,:,:]
        
        return original_img, shadow_mask
        
    def __len__(self):
        return len(self.test_set_path)
