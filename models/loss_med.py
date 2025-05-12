
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from models.loss_ssim import ssim

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1) #水平梯度
        sobely=F.conv2d(x, self.weighty, padding=1) #垂直梯度
        return torch.abs(sobelx)+torch.abs(sobely) #得到边缘响应

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        print('image_fused', image_fused.shape)
        print('intensity_joint',intensity_joint.shape)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

#新加cc loss
def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

class L_cc(nn.Module):
    def __init__(self):
        super(L_cc, self).__init__()

    def forward(self,img1_B, img2_B,img1_D,img2_D):
        cc_loss_B = cc(img1_B, img2_B)
        cc_loss_D = cc(img1_D, img2_D)
        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        return loss_decomp

class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.L_CC = L_cc()
    def forward(self, image_A, image_B, image_fused,A_b, B_b, A_d, B_d):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        loss_cc = 100 *self.L_CC(A_b, B_b, A_d, B_d)
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_cc

        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, loss_cc

