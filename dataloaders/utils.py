from PIL import Image
import os 
import math
import numpy as np



def read_BLUR(file_name):
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    
    img = Image.open(file_name)
    
    return img
    
    
def resizeBLUR(FILE):
    
    img = FILE.resize((320, 320))
    
    return img
    
def saveOutput( idx, prediction):

    pred = prediction.detach()
    
    path_output = '{}/{}'.format('Results','test')
    
    os.makedirs(path_output, exist_ok=True)
    
    path_save_pred = '{}/{:03d}.jpg'.format(path_output, idx)
    
    pred = pred[0, 0, :, :].data.cpu().numpy()
    pred = (pred*255.0).astype(np.uint8)
    
    pred = Image.fromarray(pred)
    
    pred.save(path_save_pred)
    