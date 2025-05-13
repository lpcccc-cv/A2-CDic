import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss, ncc_loss, fftLoss, MutualInformation, Mutual_info_reg, GradLoss
from data.brain_dataset import real_to_complex

from torchvision import models
import cv2
import torch.fft
from matplotlib import pyplot as plt
import os
import numpy as np

logger = logging.getLogger('base')


def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)

class BaseModel(BaseModel):
    def __init__(self, opt):
        super(BaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.dwt = opt["dwt"]
        self.which_dataset = opt["mode"]

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='mean').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)

                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.l1_loss = nn.L1Loss(reduction='mean').to(self.device)
            self.mi_loss = MutualInformation().to(self.device)


            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if opt['datasets']['train']['use_time'] == True and 'TMB' not in k:
                    v.requires_grad = False
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9, weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)
            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()


    def to_image(data):
        image = data.numpy()
        max_image = np.max(image)
        min_image = np.min(image)
        gap = max_image - min_image
        image = (image - min_image)/gap*255.
        return image
    
    def feed_data(self, data):   

        self.im1_lr = data['im1_LQ'].to(self.device)
        self.im1_gt = data['im1_GT'].to(self.device)
        self.im2_lr = data['im2_LQ'].to(self.device)
        self.im2_gt = data['im2_GT'].to(self.device)
        self.mask = data['mask'].to(self.device)


    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.fake_1, self.fake_2, x_common, x_unique, y_common, y_unique, warped_y, flow = self.netG(self.im1_lr, self.im2_gt, self.mask)

        total_loss = 0
        # L1 loss
        l_pix = self.cri_pix(self.fake_1, self.im1_gt)
        l_pix_ref = self.cri_pix(self.fake_2, self.im2_gt)
        # MI loss
        l_MI = self.mi_loss(x_common, x_unique) + self.mi_loss(y_common, y_unique)
        # for reg loss
        l_rg_mi = -self.mi_loss(self.im1_gt, warped_y)

        total_loss = l_pix + 0.1*l_MI + 0.01*l_pix_ref + 0.001*l_rg_mi
        
        total_loss.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_MI'] = l_MI.item()
        self.log_dict['l_pix_ref'] = l_pix_ref.item()
        self.log_dict['l_reg_MI'] = l_rg_mi.item()



    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H,_,_,_,_,_,_,_ = self.netG(self.im1_lr, self.im2_gt)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    
    def get_current_visuals(self, need_GT=True):

        out_dict = OrderedDict()
        out_dict['im1_restore'] = self.fake_H.detach().float().cpu()
        out_dict['im1_GT'] = self.im1_gt.detach().float().cpu() 
        
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label, epoch=None):
        if epoch != None:
            self.save_network(self.netG, 'G', iter_label, str(epoch))
        else:
            self.save_network(self.netG, 'G', iter_label)
