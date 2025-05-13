from torch.utils.data import Dataset
from pathlib import Path
import pdb,numpy
import os,glob
import torch
from torchvision import transforms
from pathlib import Path
import random,cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import torch.nn as nn

def B_spline_function(flag,t):
    if (flag == 0):
        return (1 - t*t*t + 3 * t*t - 3 * t) / 6.0
    elif (flag == 1):
        return (4 + 3 * t*t*t - 6 * t*t) / 6.0
    elif (flag == 2):
        return (1 - 3 * t*t*t + 3 * t*t + 3 * t) / 6.0
    elif (flag == 3):
        return (t*t*t / 6.0)
    else:
        return 0.0

def B_spline_transform(srcImg):
    delta_x = 32
    delta_y = 32
    H,W,C = srcImg.shape
    dsrImg =  np.zeros((H,W,C),dtype='uint8')
    grid_rows = (int)(H / delta_x) + 1 + 3
    grid_cols = (int)(W / delta_y) + 1 + 3
    noiseMat = np.zeros((grid_rows,grid_cols,2))
    offset = np.zeros((H,W,2))

    for i in range(grid_rows):
        for j in range(grid_cols):
            for k in range(2):
                noiseMat[i,j,k] = random.randint(-10,10)

    #B_spline 变形
    for x in range(H):
        for y in range(W):

            i = int(x / delta_x)  #int
            j = int(y / delta_y)

            u = float(x / delta_x - i)  #float
            v = float(y / delta_y - j)

            px = [0 for n in range(4)]
            py = [0 for n in range(4)]

            for k in range(4):
                px[k] = float(B_spline_function(k,u))  #float
                py[k] = float(B_spline_function(k,v))
            
            Tx = 0
            Ty = 0
            for m in range(4):
                for n in range(4):
                    control_point_x = int(i + m)  #int
                    control_point_y = int(j + n)
                    temp = float(py[n] * px[m])

                    Tx += temp * noiseMat[control_point_x,control_point_y,0]
                    Ty += temp * noiseMat[control_point_x,control_point_y,1]
            
            offset[x,y,0] = Tx
            offset[x,y,1] = Ty
    
    #反向映射，双线性插值
    for row in range(H):
        for col in range(W):

            src_x = row + offset[row,col,0]  #float
            src_y = col + offset[row,col,1]
            x1 = int(src_x)
            y1 = int(src_y)
            x2 = int(x1 + 1)
            y2 = int(y1 + 1)

            if (x1<0 or x1>(H - 2) or y1<0 or y1>(W - 2)):
                dsrImg[row,col,0] = 0
                dsrImg[row,col,1] = 0
                dsrImg[row,col,2] = 0
            else:
                pointa = []
                pointb = []
                pointc = []
                pointd = []

                pointa = srcImg[x1,y1,:]
                pointb = srcImg[x2,y1,:]
                pointc = srcImg[x1,y2,:]
                pointd = srcImg[x2,y2,:]

                B = (int)((x2 - src_x)*(y2 - src_y)*pointa[0] - (x1 - src_x)*(y2 - src_y)*pointb[0] - (x2 - src_x)*(y1 - src_y)*pointc[0] + (x1 - src_x)*(y1 - src_y)*pointd[0])
                G = (int)((x2 - src_x)*(y2 - src_y)*pointa[1] - (x1 - src_x)*(y2 - src_y)*pointb[1] - (x2 - src_x)*(y1 - src_y)*pointc[1] + (x1 - src_x)*(y1 - src_y)*pointd[1])
                R = (int)((x2 - src_x)*(y2 - src_y)*pointa[2] - (x1 - src_x)*(y2 - src_y)*pointb[2] - (x2 - src_x)*(y1 - src_y)*pointc[2] + (x1 - src_x)*(y1 - src_y)*pointd[2])

                dsrImg[row,col,0] = B
                dsrImg[row,col,1] = G
                dsrImg[row,col,2] = R

                # dsrImg.dtype = np.

    return dsrImg
    



class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


img_file = 'input file path'
save_file = 'save file path'
file_name_list = sorted(os.listdir(img_file))

for name in file_name_list:
    if name in os.listdir(save_file):
        continue
    img = cv2.imread(os.path.join(img_file, name))
    img = B_spline_transform(img)
    cv2.imwrite(os.path.join(save_file, name), img)
    print(name)


