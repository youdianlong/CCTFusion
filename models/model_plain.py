from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
import os

from torch.utils.tensorboard import SummaryWriter
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        
        self.netG = define_G(opt) #
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval() #设置为评估模式
        # ------------------------------------
        # Define Tensorboard 
        # ------------------------------------
        tensorboard_path = os.path.join(self.opt['path']['root'], 'Tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
        

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()
            print('end the load')
    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type'] #med

        if G_lossfn_type == 'med':
            from models.loss_med import fusion_loss_med,L_cc
            self.G_lossfn = fusion_loss_med().to(self.device)
        elif G_lossfn_type == 'vif':
            from models.loss_vif import fusion_loss_vif ,L_cc#红外loss
            self.G_lossfn = fusion_loss_vif().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    def feed_data(self, data, need_GT=False, phase='test'): #将网络加载到cuda中训练
        self.A = data['A'].to(self.device)
        self.B = data['B'].to(self.device)
        if need_GT:
            self.GT = data['GT'].to(self.device)
        print('self.device',self.device)
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, phase='test'):
        G_lossfn_type = self.opt_train['G_lossfn_type'] #med

        self.E, self.A_b, self.B_b, self.A_d, self.B_d = self.netG(self.A, self.B)  # 2.得到模型中的数值，self.E为融合后的图像,后接训练出的中间值
        self.E.to(self.device)
        self.A_b.to(self.device)
        self.B_b.to(self.device)
        self.A_d.to(self.device)
        self.B_d.to(self.device)
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_lossfn_type = self.opt_train['G_lossfn_type']
        ## constructe loss function

        if G_lossfn_type in ['vif','mef', 'mff',  'nir', 'med']:
            total_loss, loss_text, loss_int, loss_ssim,loss_cc = self.G_lossfn(self.A, self.B, self.E,self.A_b, self.B_b, self.A_d, self.B_d) #3.优化器调用损失，传入数据进行计算
            G_loss = self.G_lossfn_weight * total_loss

        else:
            total_loss, loss_text, loss_int, loss_ssim = self.G_lossfn(self.A, self.B, self.E)  # 3.优化器调用损失，传入数据进行计算
            G_loss = self.G_lossfn_weight * total_loss
        G_loss.backward()

        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item()
        if G_lossfn_type in ['loe', 'mef', 'vif', 'mff', 'gt', 'nir', 'med']:
            self.log_dict['Text_loss'] = loss_text.item()
            self.log_dict['Int_loss'] = loss_int.item()
            self.log_dict['SSIM_loss'] = loss_ssim.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])
        # ----------------------------------------
        # write tensorboard
        # ----------------------------------------
        if G_lossfn_type == 'mef':
            self.writer.add_image('under_image[0]', self.A[0])
            self.writer.add_image('under_image[1]', self.A[-1])
            self.writer.add_image('over_image[0]', self.B[0])
            self.writer.add_image('over_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'vif':
            self.writer.add_image('ir_image[0]', self.A[0])
            self.writer.add_image('ir_image[1]', self.A[1])
            self.writer.add_image('vi_image[0]', self.B[0])
            self.writer.add_image('vi_image[1]', self.B[1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'mff':
            self.writer.add_image('near_image[0]', self.A[0])
            self.writer.add_image('near_image[1]', self.A[-1])
            self.writer.add_image('far_image[0]', self.B[0])
            self.writer.add_image('far_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'nir':
            self.writer.add_image('Nir_image[0]', self.A[0])
            self.writer.add_image('Nir_image[1]', self.A[-1])
            self.writer.add_image('RGB_image[0]', self.B[0])
            self.writer.add_image('RGB_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'med':
            self.writer.add_image('pet_image[0]', self.A[0])
            self.writer.add_image('pet_image[1]', self.A[-1])
            self.writer.add_image('MRI_image[0]', self.B[0])
            self.writer.add_image('MRI_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward(phase='test')
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=False):
        out_dict = OrderedDict()
        out_dict['A'] = self.A.detach()[0].float().cpu()
        out_dict['B'] = self.B.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['GT'] = self.GT.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['A'] = self.A.detach().float().cpu()
        out_dict['BL'] = self.B.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['GT'] = self.GT.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        # print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        # print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
