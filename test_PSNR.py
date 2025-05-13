import torch
from utils import util
from tqdm import tqdm
import cv2, os
import models.modules.CDic_Align_l as CDic_Align_l
import models.modules.CDic_Align_s as CDic_Align_s
import numpy as np

import time

def main():
    mode = 'IXI'
    dataset_opt = {}
    dataset_opt['task'] = 'sr'
    dataset_opt['scale'] = 8
    dataset_opt['hr_in'] = True
    dataset_opt['crop_size'] = 0
    dataset_opt['test_size'] = 256
    
    # dataset_opt['align'] = False
    dataset_opt['align'] = True
    
    save_result = True

    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    #### create train and val dataloader
    if mode == 'IXI':
        from data.IXI_dataset import IXI_train as D
        dataset_opt['dataroot_GT'] = '~/test/T2'
    elif mode == 'fastmri':
        from data.knee_dataset import knee_train as D
        dataset_opt['dataroot_GT'] = '~/test/PDFS'
    elif mode == 'brain':
        from data.brain_dataset import brain_train as D
        dataset_opt['dataroot_GT'] = '~/T2_test'
    
    val_set = D(dataset_opt, train=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))  

    model_path = '/~.pth'
    model = CDic_Align_l.CDic_Align().cuda()
    # model = CDic_Align_s.CDic_Align().cuda()
   
    
    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        #### validation
          
        avg_psnr_im1 = 0.0
        avg_ssim_im1 = 0.0
        avg_rmse_im1 = 0.0
        idx = 0
        psnr = []
        ssim = []
        rmse = []

        for i,val_data in enumerate(tqdm(val_loader)): 

            im1_lr = val_data['im1_LQ'].cuda()
            im1_gt = val_data['im1_GT'].cuda()
            im2_lr = val_data['im2_LQ'].cuda()
            im2_gt = val_data['im2_GT'].cuda()
            mask = val_data['mask'].cuda()

            sr_img_1,_,_,_,_,_,_,_ = model(im1_lr, im2_gt)
            sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
            # sr_img_1 = im1_lr[0,0].cpu().detach().numpy()*255.
            im1_gt = im1_gt[0,0].cpu().detach().numpy()*255.

            # calculate PSNR
            cur_psnr_im1 = util.calculate_psnr(sr_img_1, im1_gt)
            psnr.append(cur_psnr_im1)
            avg_psnr_im1 += cur_psnr_im1
            cur_ssim_im1 = util.calculate_ssim(sr_img_1, im1_gt)
            ssim.append(cur_ssim_im1)
            avg_ssim_im1 += cur_ssim_im1
            cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt)
            rmse.append(cur_rmse_im1)
            avg_rmse_im1 += cur_rmse_im1
        

            # 保存图像
            # save_path_1 = '~/SR_result/'+ model_name
            # if not os.path.exists(save_path_1):
            #     os.makedirs(save_path_1)
            # if save_result:
            #     cv2.imwrite(os.path.join(save_path_1, '{:08d}.png'.format(i+1)), sr_img_1)

            idx += 1
        
        avg_psnr_im1 = avg_psnr_im1 / idx
        avg_ssim_im1 = avg_ssim_im1 / idx
        avg_rmse_im1 = avg_rmse_im1 / idx

        # log
        std_psnr = np.sqrt(np.mean((psnr - np.mean(psnr)) ** 2))
        std_ssim = np.sqrt(np.mean((ssim - np.mean(ssim)) ** 2))
        std_rmse = np.sqrt(np.mean((rmse - np.mean(rmse)) ** 2))
        print("# image1 Validation # PSNR: {:.6f}, std: {:.5f}".format(avg_psnr_im1, std_psnr))
        print("# image1 Validation # SSIM: {:.6f}, std: {:.5f}".format(avg_ssim_im1, std_ssim))
        print("# image1 Validation # RMSE: {:.6f}, std: {:.5f}".format(avg_rmse_im1, std_rmse))


### CUDA_VISIBLE_DEVICES=2 python test_PSNR.py
if __name__ == '__main__':
    main()

