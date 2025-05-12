import argparse
from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys
from models.network_cctfusion import CCTFusion as net
from torchvision import transforms

from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
from models.img_read_save import img_save,image_read_cv2
import warnings
warnings.filterwarnings("ignore")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/Model/Medical_Fusion-CT-MRI/Medical_Fusion/models/')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    parser.add_argument('--root_path', type=str, default='./Dataset/test_img/',
                        help='input test image root folder')
    parser.add_argument('--save_path', type=str, default='./results/test_tmp_result/',
                        help='output test image root folder')
    parser.add_argument('--image_size', type=int, default=256,help='input image size')

    args = parser.parse_args()

    for dataset_name in ["MRI_CT", "MRI_PET", "MRI_SPECT"]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # set up model
        model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
        if os.path.exists(model_path):
            print(f'loading model from {args.model_path}')
        else:
            print('Traget model path: {} not existing!!!'.format(model_path))
            sys.exit()
        model = define_model(args)
        model.eval()
        model = model.to(device)

        # setup folder and path
        folder, save_dir, border, window_size = setup(args,dataset_name)
        a_dir = os.path.join(args.root_path, dataset_name, dataset_name.split('_')[0])
        b_dir = os.path.join(args.root_path, dataset_name, dataset_name.split('_')[1])

        os.makedirs(save_dir, exist_ok=True)
        test_set = D(a_dir, b_dir, args.in_channel,args.image_size)

        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 drop_last=False, pin_memory=True)
        for i, test_data in enumerate(test_loader):
            imgname = test_data['A_path'][0]
            img_a = test_data['A'].to(device)
            img_b = test_data['B'].to(device)
            sizes = img_a.size()
            H, W = sizes[2], sizes[3]
            start = time.time()
            # inference
            with torch.no_grad():
                output = test(img_a, img_b, model, args, window_size)
            end = time.time()
            output = util.tensor2uint(output)
            save_name = os.path.join(save_dir, os.path.basename(imgname))
            util.imsave(output, save_name)

def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=args.image_size, window_size=8,#args.image_size
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


def setup(args,dataset_name):
    save_dir = f'results/temp/CCTFusion_{dataset_name}'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    border = 0
    window_size = 8
    return folder, save_dir, border, window_size

if __name__ == '__main__':
    main()
