import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

from model.mobilenet import MobileNetV3_backbone
from timm.models.layers import trunc_normal_

MIN_NUM_PATCHES=15

class face_landmark_4simmin_glo_loc(nn.Module):
    def __init__(self, *, loss_type, image_size, patch_size, dim,  pool = 'cls',num_patches=None, channels = 3, emb_dropout = 0.,fp16=True):
        super().__init__()

        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # refer to paper section 6
            nn.Linear(160, self.row_num * self.row_num * 2),  # 2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),  # refer to paper section 6
            nn.Linear(160, dim),  #
        )
            # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
            # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape = torch.tensor([patch_size, patch_size])  # 196
        self.theta = 0

        self.drop_2d = torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.loss_type = loss_type
        self.sigmoid = nn.Sigmoid()
        self.num_features = dim
        self.in_chans = channels

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, x_Aug=None,keep_num=None,patch_shape=torch.tensor([10,10]),Random_prob=False,return_prob=False,ran_sample=False,random_coor=False,return_land=False):
        p = self.patch_size
        if not random_coor:
            # get the image shape
            imgshape = x.shape[-2]
            if self.num_patches == 144 and imgshape == 112:
                keep_num = self.num_patches
            elif self.num_patches == 196 and imgshape == 112:
                keep_num = self.num_patches
            else:
                keep_num = (imgshape // p) ** 2

            theta = self.stn(x)  # .forward(x)            #with original stn

            theta0 = theta.mean(dim=(-2, -1))  # average pooling   for cnn
            theta = self.output_layer(theta0)
            glo_token = self.global_token(theta0).view(-1, 1, self.dim)

            t_max = torch.max(theta, 1)[0]  # .repeat(1,49*2)
            t_max = torch.unsqueeze(t_max, dim=1).repeat(1, self.row_num * self.row_num * 2)
            t_min = torch.min(theta, 1)[0]  # .repeat(1,49*2)
            t_min = torch.unsqueeze(t_min, dim=1).repeat(1, self.row_num * self.row_num * 2)
            theta = (theta - t_min) / (t_max - t_min) * 111

            theta = theta.view(-1, self.row_num * self.row_num, 2)
            if Random_prob:
                prob = torch.randn(theta.shape) * 5  # *12# 10 pixel
                theta = theta + prob.cuda()
                if not return_prob:
                    b, c, fea = theta.shape
                    if ran_sample:
                        extract_id = torch.randint(0, c, (b, 36, 1)).cuda()
                        keep_num = 36
                    else:
                        extract_id = torch.randint(0, c, (b, self.row_num * self.row_num, 1)).cuda()
                        keep_num = self.row_num * self.row_num
                    extract_id = extract_id.repeat(1, 1, 2)
                    # extract landmarks
                    out_theta = torch.gather(theta, 1, extract_id)
                    theta = out_theta
            self.theta = theta  # .detach()

            if keep_num is not None:
                num_land = keep_num
            else:
                num_land = theta.shape[-2]  # int(self.row_num*self.row_num)
        else:
            b = x.shape[0]
            if not ran_sample:
                num_land = self.row_num * self.row_num
            else:
                num_land = 25
            new_theta = torch.rand(b, num_land, 2) * 111.0  # .cuda()
            theta = new_theta.cuda()
        if return_land:
            return theta, x
        else:
            if x_Aug is None:
                x = extract_patches_pytorch_gridsample(x, theta[:, :num_land], patch_shape=self.patch_shape,
                                                       num_landm=num_land)
            else:
                x = extract_patches_pytorch_gridsample(x_Aug, theta[:, :num_land], patch_shape=self.patch_shape,
                                                       num_landm=num_land)

            return theta, x


def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape, num_landm=49):
    device = landmarks.device
    img_shape = imgs.shape[2]
    list_patches = []
    patch_half_shape = patch_shape / 2
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = torch.meshgrid(torch.arange(start[0], end[0]),
                                   torch.arange(start[1], end[1]))  # start[0]:end[0], start[1]:end[1]]
    sampling_grid = torch.stack(sampling_grid, dim=0).to(device)  # .cuda()
    sampling_grid = torch.transpose(torch.transpose(sampling_grid, 0, 2), 0, 1)
    for i in range(num_landm):
        land = landmarks[:, i, :]

        patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :]) / (img_shape * 0.5) - 1
        sing_land_patch = F.grid_sample(imgs, patch_grid, align_corners=False)
        list_patches.append(sing_land_patch)
    list_patches = torch.stack(list_patches, dim=2)  # .shape
    B, c, patches_num, w, h = list_patches.shape
    row = int(np.sqrt(patches_num))
    list_patches = list_patches.reshape(B, c, row, row, w, h)
    list_patches = list_patches.permute(0, 1, 2, 4, 3, 5)
    list_patches = list_patches.reshape(B, c, w * int(np.sqrt(patches_num)), h * int(np.sqrt(patches_num)))
    
    return list_patches