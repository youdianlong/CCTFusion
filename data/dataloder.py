import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image
from models.img_read_save import img_save,image_read_cv2
import cv2


class Dataset(data.Dataset): 

    def __init__(self, root_A, root_B, in_channels,image_size):
        super(Dataset, self).__init__()
        self.paths_A = util.get_image_paths(root_A)
        self.paths_B = util.get_image_paths(root_B)
        self.inchannels = in_channels
        self.image_size = (image_size, image_size)

    def __getitem__(self, index):

        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, 1,self.image_size)
        img_B = util.imread_uint(B_path, self.inchannels,self.image_size)#

        img_A = util.uint2single(img_A)
        img_B = util.uint2single(img_B)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = util.single2tensor3(img_A)
        img_B = util.single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
