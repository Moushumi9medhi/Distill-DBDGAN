"""
*****************************************

Distill-DBDGAN:
paths_and_transform
*****************************************
*****************************************

DEVELOPER: SANKARAGANESH JONNA, MOUSHUMI MEDHI

*****************************************
"""
import glob
import os
import os.path
import torchvision.transforms as transforms

def blur_dataset(args):
    dataset = []
    original_img_rpath = args.blurImg_path 
    shadow_mask_rpath = args.blurmap_path 
    for img_path in glob.glob(os.path.join(original_img_rpath, '*.jpg')):
        basename = os.path.basename(img_path)
        original_img_path = os.path.join(original_img_rpath, basename)
        shadow_mask_path = os.path.join(shadow_mask_rpath, basename)
        dataset.append([original_img_path, shadow_mask_path])
    return dataset
    
def SZU_dataset(args):
    dataset = []
    original_img_rpath = args.blurImg_path 
    shadow_mask_rpath = args.blurmap_path 
    for img_path in glob.glob(os.path.join(original_img_rpath, '*.jpg')):
        basename = os.path.basename(img_path)
        original_img_path = os.path.join(original_img_rpath, basename)
        shadow_mask_path = os.path.join(shadow_mask_rpath, os.path.splitext(basename)[0] + '.png' )
        dataset.append([original_img_path, shadow_mask_path])
    return dataset
    



def blurdatatransforms(img):
    img = transforms.ToTensor(img)
    return img