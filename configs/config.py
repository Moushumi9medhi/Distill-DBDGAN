from yacs.config import CfgNode as CN

cfg = CN()

# Hardware
cfg.seed = 42
cfg.gpus = (1, )

cfg.use_gpu = True
cfg.test = 1
cfg.cudnn_deterministic = False
cfg.cudnn_benchmark = False
cfg.num_threads = 1
cfg.test_option = ''
cfg.batch_size = ''
cfg.shuffle = ''
cfg.num_workers = ''
cfg.test_model_path = ''
cfg.blurImg_path = '' 
cfg.blurmap_path = '' 
cfg.save_test_image = False





def get_cfg_defaults():
    
    return cfg.clone()


if __name__ == '__main__':
    my_cfg = get_cfg_defaults()
    print(my_cfg)
