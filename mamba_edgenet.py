"""Training and evaluation code for Mamba-EdgeNet.

This public-release file preserves the implementation used for the paper
"Mamba-EdgeNet: Learnable Edge-Guided State Space Model for Skin Lesion
Segmentation".
"""

import argparse
import gc
import logging
import math
import os
import random
import sys
import time
from typing import Dict, Optional

import albumentations as albu
import cv2
import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from mamba_ssm.modules.mamba2_simple import Mamba2Simple

from ef_albp_torch import albp_torch, edge_aware_filtering_torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed=2026):
    """Configure the random-number generators used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """Initialize a data-loader worker with a deterministic seed."""
    seed = 2026 + worker_id
    np.random.seed(seed)
    random.seed(seed)
g = torch.Generator()
g.manual_seed(2026)
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/isic2016', help='Dataset root path')
parser.add_argument('--exp', type=str, default='MambaEdgeNet', help='Experiment name')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--model', type=str, default='res50_mamba', help='Model name')
parser.add_argument('--max_iterations', type=int, default=10000, help='Maximum training iterations')
parser.add_argument('--batch_size', type=int, default=6, help='Batch size per GPU')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0003, help='Base learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='Input patch size')
parser.add_argument('--seed', type=int, default=2026, help='Random seed')
parser.add_argument('--labeled_num', type=int, default=140, help='Number of labeled samples')
parser.add_argument('--input_h', type=int, default=256, help='Input height')
parser.add_argument('--input_w', type=int, default=256, help='Input width')
parser.add_argument('--mode', choices=('train', 'test'), default='test', help='Run training before evaluation, or evaluate an existing checkpoint')
parser.add_argument('--output_dir',type=str,default='./outputs',help='Directory for checkpoints and results')
args = parser.parse_args()

class MultiDirectionMamba2DBlock(nn.Module):

    def __init__(self, dim, headdim=16):
        super().__init__()
        self.row_mamba_fwd = Mamba2Simple(d_model=dim, headdim=headdim)
        self.row_mamba_bwd = Mamba2Simple(d_model=dim, headdim=headdim)
        self.col_mamba_fwd = Mamba2Simple(d_model=dim, headdim=headdim)
        self.col_mamba_bwd = Mamba2Simple(d_model=dim, headdim=headdim)
        num_groups = 8 if dim % 8 == 0 else 1
        self.norm_branches = nn.ModuleList([nn.GroupNorm(num_groups, dim) for _ in range(4)])
        self.norm_fuse = nn.LayerNorm(4 * dim)
        self.fuse = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        dtype_orig = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, C, H, W = x.shape
            x_r = x.permute(0, 2, 3, 1).reshape(B * H, W, C).contiguous()
            out_r_fwd = self.row_mamba_fwd(x_r).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            out_r_fwd = self.norm_branches[0](out_r_fwd)
            out_r_bwd = self.row_mamba_bwd(x_r.flip(dims=[1]).contiguous()).flip(dims=[1]).contiguous()
            out_r_bwd = out_r_bwd.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            out_r_bwd = self.norm_branches[1](out_r_bwd)
            x_c = x.permute(0, 3, 2, 1).reshape(B * W, H, C).contiguous()
            out_c_fwd = self.col_mamba_fwd(x_c).view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            out_c_fwd = self.norm_branches[2](out_c_fwd)
            out_c_bwd = self.col_mamba_bwd(x_c.flip(dims=[1]).contiguous()).flip(dims=[1]).contiguous()
            out_c_bwd = out_c_bwd.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            out_c_bwd = self.norm_branches[3](out_c_bwd)
            out = torch.cat([out_r_fwd, out_r_bwd, out_c_fwd, out_c_bwd], dim=1)
            out = out.permute(0, 2, 3, 1)
            out = self.norm_fuse(out).permute(0, 3, 1, 2)
            out = self.fuse(out) + x
        return out.to(dtype_orig)

class EfficientMamba2DBlock(nn.Module):
    """Apply resolution-adaptive two-dimensional Mamba2 scanning."""

    def __init__(self, channels=None, dim=None, mamba_dim=None, hw_threshold=16384, headdim=16, mamba_kwargs=None):
        super().__init__()
        if mamba_kwargs is None:
            mamba_kwargs = {}
        actual_in_channels = channels if channels is not None else dim
        if actual_in_channels is None:
            raise ValueError("Must provide either 'channels' or 'dim'")
        actual_mamba_dim = mamba_dim if mamba_dim is not None else actual_in_channels
        self.hw_threshold = hw_threshold
        if actual_in_channels != actual_mamba_dim:
            self.proj_in = nn.Sequential(nn.Conv2d(actual_in_channels, actual_mamba_dim, 1, bias=False), nn.BatchNorm2d(actual_mamba_dim), nn.SiLU(inplace=True))
            self.proj_out = nn.Conv2d(actual_mamba_dim, actual_in_channels, kernel_size=1, bias=False)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        self.norm = nn.BatchNorm2d(actual_in_channels)
        self.mamba_lowres = MultiDirectionMamba2DBlock(actual_mamba_dim, headdim=headdim)
        self.mamba_row_seq = Mamba2Simple(d_model=actual_mamba_dim, headdim=headdim, **mamba_kwargs)
        self.mamba_col_seq = Mamba2Simple(d_model=actual_mamba_dim, headdim=headdim, **mamba_kwargs)
        self.norm_mid = nn.LayerNorm(actual_mamba_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = self.proj_in(x)
        B, C, H, W = x.shape
        if H * W <= self.hw_threshold:
            out = self.mamba_lowres(x)
        else:
            row_seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C).contiguous()
            row_out = self.mamba_row_seq(row_seq).reshape(B, H, W, C)
            row_out = self.norm_mid(row_out)
            col_seq = row_out.permute(0, 2, 1, 3).reshape(B * W, H, C).contiguous()
            col_out = self.mamba_col_seq(col_seq)
            out = col_out.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return identity + self.proj_out(out)

class Res50MambaEncoder(nn.Module):

    def __init__(self, in_chans=3, ft_chns=[16, 32, 64, 128, 256], pretrained=True, pretrained_path=None):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = resnet50(weights=weights)
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.resnet.load_state_dict(state_dict, strict=False)
        self.resnet.conv1.stride = (1, 1)
        self.in_conv_block = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        res_chs = [64, 256, 512, 1024, 2048]
        self.project_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(res_chs[i], ft_chns[i], kernel_size=1, bias=False), nn.BatchNorm2d(ft_chns[i]), nn.ReLU(inplace=True)) for i in range(5)])
        self.mamba3 = EfficientMamba2DBlock(res_chs[3], mamba_dim=ft_chns[3])
        self.mamba4 = EfficientMamba2DBlock(res_chs[4], mamba_dim=ft_chns[4])

    def in_conv(self, x):
        return self.in_conv_block(x)

    def down1(self, x):
        return self.project_convs[1](self.layer1(self.maxpool(x)))

    def forward(self, x):
        x0 = self.in_conv_block(x)
        x1 = self.project_convs[0](x0)
        m = self.maxpool(x0)
        l1 = self.layer1(m)
        x2 = self.project_convs[1](l1)
        l2 = self.layer2(l1)
        x3 = self.project_convs[2](l2)
        l3 = self.mamba3(self.layer3(l2))
        x4 = self.project_convs[3](l3)
        l4 = self.mamba4(self.layer4(l3))
        x5 = self.project_convs[4](l4)
        return [x1, x2, x3, x4, x5]

class UpBlock1(nn.Module):

    def __init__(self, in_chns, cat_chns, out_chns, dropout_p=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_chns + cat_chns, out_chns, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_chns), nn.ReLU(inplace=True), nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(), nn.Conv2d(out_chns, out_chns, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_chns), nn.ReLU(inplace=True))

    def forward(self, x, concat_features):
        x_up = self.upsample(x)
        if x_up.shape[2:] != concat_features.shape[2:]:
            x_up = F.interpolate(x_up, size=concat_features.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x_up, concat_features], dim=1))

class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']
        self.dropout = params['dropout']
        self.up1 = UpBlock1(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[0])
        self.up2 = UpBlock1(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[1])
        self.up3 = UpBlock1(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[2])
        self.up4 = UpBlock1(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[3])
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out_conv(x)

class UNet_Res50Mamba(nn.Module):

    def __init__(self, in_chns=3, class_num=2, ft_chns=[16, 32, 64, 128, 256], res50_encoder=None):
        super().__init__()
        if res50_encoder is None:
            raise RuntimeError('UNet_Res50Mamba requires a res50_encoder instance.')
        self.encoder = res50_encoder
        params = {'in_chns': in_chns, 'feature_chns': ft_chns, 'class_num': class_num, 'bilinear': False, 'dropout': [0.05, 0.1, 0.1, 0.1, 0.1]}
        self.decoder = Decoder(params)
        self.num_classes = class_num

    def forward(self, x):
        return self.decoder(self.encoder(x))

class ISIC2016Dataset(Dataset):
    """Load ISIC 2016 images and binary masks.

    Expected layout: ``data_root/{split}/data`` for JPEG images and
    ``data_root/{split}/mask`` for ``*_Segmentation.png`` masks.
    """

    def __init__(self, data_root: str, split: str='train', transform: Optional[albu.Compose]=None):
        self.data_path = os.path.join(data_root, split, 'data')
        self.mask_path = os.path.join(data_root, split, 'mask')
        self.transform = transform
        self.image_files = sorted((f for f in os.listdir(self.data_path) if f.endswith('.jpg') and os.path.exists(os.path.join(self.mask_path, f.replace('.jpg', '_Segmentation.png')))))
        self.cache = {}
        self.cache_size = 20

    def __len__(self):
        """Return number of images in dataset"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        with Image.open(os.path.join(self.data_path, img_name)) as img:
            img = img.convert('RGB')
            img = np.array(img)
        with Image.open(os.path.join(self.mask_path, f'{base_name}_Segmentation.png')) as mask:
            mask = np.array(mask.convert('L'))
            mask = (mask > 128).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = (augmented['image'], augmented['mask'])
        assert img.shape[-1] == 3, f'Image should have 3 channels, got {img.shape[-1]}'
        result = {'image': torch.from_numpy(img).permute(2, 0, 1).float(), 'label': torch.from_numpy(mask).long()}
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = result
        return result

def get_train_transform():
    """Get training transformations"""
    return albu.Compose([albu.Resize(256, 256), albu.ToFloat(max_value=255.0), albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0), albu.OneOf([albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5), albu.RandomRotate90(p=0.5)], p=0.8)], p=1.0)

def get_val_transform():
    """Get validation transformations (no augmentation)"""
    return albu.Compose([albu.Resize(256, 256), albu.ToFloat(max_value=255.0), albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)])

class ISIC2017Dataset(Dataset):
    """Load ISIC 2017 from ``../data/isic2017``.

    The loader expects the retained Training, Validation, and Test Data and
    GroundTruth directory names used by the experiment.
    """

    def __init__(self, split='train', size=512, transform: Optional[albu.Compose]=None, cache_size: int=20):
        base_dir = '../data/isic2017'
        if split == 'train':
            image_root = os.path.join(base_dir, 'ISIC2017_Training_Data')
            mask_root = os.path.join(base_dir, 'ISIC2017_Training_GroundTruth')
        elif split == 'val':
            image_root = os.path.join(base_dir, 'ISIC2017_Validation_Data')
            mask_root = os.path.join(base_dir, 'ISIC2017_Validation_GroundTruth')
        elif split == 'test':
            image_root = os.path.join(base_dir, 'ISIC2017_Test_Data')
            mask_root = os.path.join(base_dir, 'ISIC2017_Test_GroundTruth')
        else:
            raise ValueError(f'Unknown split: {split}')
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg')) and '_superpixels' not in f])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.images) == len(self.masks), f'Image/mask count mismatch: images={len(self.images)}, masks={len(self.masks)}'
        self.size = size
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = (augmented['image'], augmented['mask'])
            result = {'image': torch.from_numpy(image).permute(2, 0, 1).float(), 'label': torch.from_numpy(mask).float()}
        else:
            image = cv2.resize(image, (self.size, self.size)).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (self.size, self.size))
            mask = (mask > 0.5).astype(np.float32)
            result = {'image': torch.from_numpy(image).permute(2, 0, 1).float(), 'label': torch.from_numpy(mask).unsqueeze(0).float()}
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = {k: v.clone() for k, v in result.items()}
        return result

class ISIC2018Dataset(Dataset):
    """Load ISIC 2018 from ``../data/isic2018``.

    The loader expects the retained Training, Validation, and Test Input and
    GroundTruth directory names used by the experiment.
    """

    def __init__(self, split='train', size=512, transform: Optional[albu.Compose]=None):
        """Initialize an ISIC 2018 split at the retained resize setting."""
        base_dir = '../data/isic2018'
        if split == 'train':
            image_root = os.path.join(base_dir, 'ISIC2018_Training_Input')
            mask_root = os.path.join(base_dir, 'ISIC2018_Training_GroundTruth')
        elif split == 'val':
            image_root = os.path.join(base_dir, 'ISIC2018_Validation_Input')
            mask_root = os.path.join(base_dir, 'ISIC2018_Validation_GroundTruth')
        elif split == 'test':
            image_root = os.path.join(base_dir, 'ISIC2018_Test_Input')
            mask_root = os.path.join(base_dir, 'ISIC2018_Test_GroundTruth')
        else:
            raise ValueError(f'Unknown split: {split}')
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.size = size
        assert len(self.images) == len(self.masks), f'Image/mask count mismatch: images={len(self.images)}, masks={len(self.masks)}'
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = (augmented['image'], augmented['mask'])
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()
        return {'image': image, 'label': mask}

def get_isic2018_loader(split='train', batchsize=6, size=512, shuffle=True, num_workers=4, pin_memory=True, transform: Optional[albu.Compose]=None):
    """Build a data loader for the configured ISIC 2018 split."""
    dataset = ISIC2018Dataset(split=split, size=size, transform=transform)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle if split == 'train' else False, num_workers=num_workers, pin_memory=pin_memory)
    return loader

class PH2Dataset(Dataset):
    """Load PH2 from ``base_dir/images`` and ``base_dir/masks``.

    Masks must use the ``*_Segmentation.png`` naming convention.
    """

    def __init__(self, base_dir='../data/PH2', size=512, transform: Optional[albu.Compose]=None, cache_size: int=20):
        self.image_root = os.path.join(base_dir, 'images')
        self.mask_root = os.path.join(base_dir, 'masks')
        self.images = sorted([os.path.join(self.image_root, f) for f in os.listdir(self.image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = []
        for img_path in self.images:
            file = os.path.basename(img_path).split('.')[0]
            mask_path = os.path.join(self.mask_root, f'{file}_Segmentation.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'Mask not found: {mask_path}')
            self.masks.append(mask_path)
        assert len(self.images) == len(self.masks), f'Image/mask count mismatch: images={len(self.images)}, masks={len(self.masks)}'
        self.size = size
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = (augmented['image'], augmented['mask'])
        else:
            image = cv2.resize(image, (self.size, self.size)).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (self.size, self.size)).astype(np.float32)
        result = {'image': torch.from_numpy(image).permute(2, 0, 1).float(), 'label': torch.from_numpy(mask).unsqueeze(0).float()}
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = {k: v.clone() for k, v in result.items()}
        return result

class PH2SplitDataset(Dataset):
    """Load the retained PH2 train/validation layout.

    The loader expects ``PH2_Train_Data``, ``PH2_Train_GroundTruth``,
    ``PH2_Val_Data``, and ``PH2_Val_GroundTruth`` under ``base_dir``.
    """

    def __init__(self, base_dir='../data/PH2', split='train', size=512, transform: Optional[albu.Compose]=None, cache_size=20):
        assert split in ['train', 'val'], f'Unknown split: {split}'
        if split == 'train':
            self.image_root = os.path.join(base_dir, 'PH2_Train_Data')
            self.mask_root = os.path.join(base_dir, 'PH2_Train_GroundTruth')
        else:
            self.image_root = os.path.join(base_dir, 'PH2_Val_Data')
            self.mask_root = os.path.join(base_dir, 'PH2_Val_GroundTruth')
        self.images = sorted([os.path.join(self.image_root, f) for f in os.listdir(self.image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = []
        for img_path in self.images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = f'{base}_Segmentation.png'
            mask_path = os.path.join(self.mask_root, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'Mask not found: {mask_path}')
            self.masks.append(mask_path)
        assert len(self.images) == len(self.masks), f'Image/mask count mismatch: images={len(self.images)}, masks={len(self.masks)}'
        self.size = size
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = (aug['image'], aug['mask'])
        else:
            img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (self.size, self.size)).astype(np.float32)
        result = {'image': torch.from_numpy(img).permute(2, 0, 1).float(), 'label': torch.from_numpy(mask).float()}
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = {k: v.clone() for k, v in result.items()}
        return result

def plot_shallow_pred_hist(shallow_pred, epoch, save_dir):
    """
    Plot histogram of shallow predictions
    Args:
        shallow_pred: Tensor [B,1,H,W] of sigmoid outputs
        epoch: Current epoch number
        save_dir: Directory to save plot
    """
    pred_np = shallow_pred.detach().cpu().numpy().flatten()
    plt.figure()
    plt.hist(pred_np, bins=50, range=(0, 1), color='skyblue', edgecolor='black')
    plt.title(f'Shallow Pred Sigmoid Histogram (Epoch {epoch})')
    plt.xlabel('Predicted Value')
    plt.ylabel('Pixel Count')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'shallow_pred_hist_epoch{epoch}.png'))
    plt.close()

def calculate_metric(pred, gt):
    """
    Calculate Dice and IoU metrics
    Args:
        pred: Prediction array
        gt: Ground truth array
    Returns:
        dice: Dice coefficient
        iou: Intersection over Union
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    intersection = np.sum(pred * gt)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-07)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-07)
    return (dice, iou)

def test_single_volume(image, label, net, classes, global_step=None, total_steps=None):
    """
    Test a single volume (image)
    Args:
        image: Input image
        label: Ground truth label
        net: Network model
        classes: Number of classes
    Returns:
        dice: Dice coefficient
        iou: Intersection over Union
    """
    net.eval()
    with torch.no_grad():
        outputs = net(image.cuda(), global_step=global_step, total_steps=total_steps)
        seg_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        output = seg_output
        if output.dim() == 4:
            pred = torch.sigmoid(output)[:, 1] > 0.5
        else:
            pred = torch.sigmoid(output) > 0.5
        pred = pred.cpu().squeeze().numpy()
    dice, iou = calculate_metric(pred, label)
    return (dice, iou, seg_output)

class ResBlock(nn.Module):
    """Residual convolutional block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""

    def __init__(self, in_channels, out_channels=64, rates=[1, 6, 12, 18]):
        super().__init__()
        self.blocks = nn.ModuleList()
        for r in rates:
            self.blocks.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.GroupNorm(num_groups=8, num_channels=out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(len(rates) * out_channels + out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        res = [block(x) for block in self.blocks]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bicubic', align_corners=False)
        res.append(gp)
        return self.project(torch.cat(res, dim=1))

class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D inputs"""

    def __init__(self, drop_prob=0.1, block_size=5):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        """Forward pass with DropBlock"""
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask.squeeze(1)
        return x * mask[:, None, :, :] * (mask.numel() / mask.sum()).clamp(min=1.0)

    def _compute_gamma(self, x):
        """Compute gamma parameter for DropBlock"""
        return self.drop_prob / self.block_size ** 2

class EnhancedEdgeModule(nn.Module):
    """Deep edge-enhancement module with Mamba2 and ASPP."""

    def __init__(self, in_channels, decoder_out_channels=64):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.mamba1 = EfficientMamba2DBlock(dim=128, headdim=16)
        self.mamba2 = EfficientMamba2DBlock(dim=128, headdim=16)
        self.refine = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.Sigmoid())
        self.dropblock = DropBlock2D(drop_prob=0.05, block_size=5)
        self.aspp = ASPP(128, 128)
        self.mamba_refine = EfficientMamba2DBlock(dim=128, headdim=8, hw_threshold=8192)
        self.dilated_refine1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dilated_refine2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, dilation=3, padding=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.edge_out = nn.Conv2d(128, 1, 1)
        self.edge_out_low = nn.Conv2d(128, 1, 1)
        if decoder_out_channels != 128:
            self.to_decoder = nn.Conv2d(128, decoder_out_channels, kernel_size=1, bias=False)
        else:
            self.to_decoder = nn.Identity()

    def forward(self, x, epoch_num=None):
        x = self.init_conv(x)
        x = self.mamba1(x)
        x = self.mamba2(x)
        refine_mask = self.refine(x)
        x = x + x * refine_mask
        f = self.aspp(x)
        f = f + self.mamba_refine(f)
        f = f + self.dilated_refine1(f)
        f = f + self.dilated_refine2(f)
        edge_pred_main = self.edge_out(f)
        edge_pred_low = self.edge_out_low(f)
        f_dec = self.to_decoder(f)
        return (edge_pred_main, edge_pred_low, f_dec)

class ShallowEdgeExtractor(nn.Module):
    """Shallow edge feature extractor"""

    def __init__(self):
        super().__init__()
        self.rgb_conv = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False), nn.Conv2d(2, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=2, dilation=2, groups=16, bias=False), nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.rgb_resblock = ResBlock(32)
        self.mamba = EfficientMamba2DBlock(dim=32, headdim=16)
        self.dropout = nn.Dropout2d(p=0.1)
        self.bimamba = EfficientMamba2DBlock(dim=32, headdim=16)
        self.local_refine = nn.Sequential(nn.Conv2d(32, 32, 3, padding=2, dilation=2, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.Dropout2d(p=0.1), nn.ReLU(inplace=True))
        self.x1_reduce = nn.Sequential(nn.Conv2d(16, 16, kernel_size=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.fuse_attention = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1), SEBlock(48, reduction=8), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(48, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), SEBlock(32))
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, ef_input, x1_feat):
        assert ef_input.dim() == 4 and ef_input.size(1) == 2, \
            f'ShallowEdgeExtractor expects [B,2,H,W], got {ef_input.shape}'

        if x1_feat is None:
            raise ValueError('x1_feat is required for shallow feature fusion.')

        feat = self.rgb_conv(ef_input)
        x1_up = F.interpolate(
            x1_feat,
            size=ef_input.shape[2:],
            mode='bicubic',
            align_corners=False
        )
        x1_proj = self.x1_reduce(x1_up)
        feat = torch.cat([feat, x1_proj], dim=1)
        feat = self.fuse_attention(feat)

        fused = self.fuse(feat)
        mamba_out = self.dropout(self.mamba(fused))
        fused = fused + mamba_out
        return self.local_refine(fused)

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class ShiftConv8(nn.Module):
    """Eight-direction shift convolution."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels * 8, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.register_buffer('shifts', torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=torch.int32))

    def forward(self, x):
        shifted = torch.cat([torch.roll(x, shifts=tuple(s), dims=(2, 3)) for s in self.shifts], dim=1)
        out = self.conv(shifted)
        return out

class LearnableEdgeEnhancer(nn.Module):

    def __init__(self, in_channels=1, sigma_s=5, sigma_r=0.5, tau=0.05, alphaT=2, debug=True):
        super(LearnableEdgeEnhancer, self).__init__()
        self.alphaT = nn.Parameter(torch.tensor(float(alphaT)))
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.sigma_s = nn.Parameter(torch.tensor(float(sigma_s)))
        self.sigma_r = nn.Parameter(torch.tensor(float(sigma_r)))
        self.debug = debug
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv8d = ShiftConv8(in_channels=32)
        self.post_shift_smooth = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.edge_scale = nn.Parameter(torch.tensor(2.0))
        self.edge_bias = nn.Parameter(torch.tensor(-1.0))
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel)
        self.register_buffer('sobel_y', sobel.transpose(2, 3))
        self.hair_removal_net = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, in_channels, 3, padding=1), nn.Sigmoid())
        self.backbone = nn.Sequential(nn.Conv2d(in_channels + 1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.register_buffer('gaussian_kernel_small', self._create_gaussian_kernel(kernel_size=5, sigma=1.0))
        self.register_buffer('gaussian_kernel_large', self._create_gaussian_kernel(kernel_size=9, sigma=2.0))
        self.edge_head = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=1))
        for m in self.edge_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, device=torch.device('cpu'))
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size).contiguous()
        return k.float()

    def compute_gradient(self, x):
        """Compute the Sobel gradient magnitude."""
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        gx = F.conv2d(x, sobel, padding=1)
        gy = F.conv2d(x, sobel.transpose(2, 3), padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        return grad_mag

    def multi_scale_filtering(self, image):
        """Apply the retained two-scale Gaussian smoothing operation."""
        small_blur = F.conv2d(image, self.gaussian_kernel_small, padding=2)
        large_blur = F.conv2d(image, self.gaussian_kernel_large, padding=4)
        return small_blur + large_blur

    def morphological_processing(self, edge_map, kernel_size=3, sharpness: float=12.0):
        """Apply differentiable soft opening while preserving edge strength."""
        x = torch.sigmoid(edge_map)
        pad = kernel_size // 2
        min_pool = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
        eroded_soft = torch.sigmoid((min_pool - 0.5) * sharpness)
        max_pool_eroded = F.max_pool2d(eroded_soft, kernel_size=kernel_size, stride=1, padding=pad)
        dilated_soft = torch.sigmoid((max_pool_eroded - 0.5) * sharpness)
        opened = dilated_soft
        out = opened * edge_map
        return out

    def soft_gradient_guided_refinement(self, edge_map, grad_mag, threshold_ratio=1.2):
        """Weight edge responses with the image-gradient magnitude."""
        mean_grad = grad_mag.mean(dim=[2, 3], keepdim=True)
        soft_threshold = mean_grad * threshold_ratio
        weight_map = torch.sigmoid((grad_mag - soft_threshold) * 5.0)
        return edge_map * weight_map

    def soft_morphological_opening(self, edge_map, kernel_size=3):
        """Apply the retained differentiable morphological opening."""
        sharpness = 2.0
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=edge_map.device)
        eroded = F.conv2d(edge_map, kernel, padding=kernel_size // 2)
        eroded_min = eroded.amin(dim=(2, 3), keepdim=True)
        eroded_max = eroded.amax(dim=(2, 3), keepdim=True)
        eroded_norm = (eroded - eroded_min) / (eroded_max - eroded_min + 1e-08)
        eroded_soft = torch.sigmoid((eroded_norm - 0.5) * sharpness)
        dilated = F.conv2d(eroded_soft, kernel, padding=kernel_size // 2)
        dilated_min = dilated.amin(dim=(2, 3), keepdim=True)
        dilated_max = dilated.amax(dim=(2, 3), keepdim=True)
        dilated_norm = (dilated - dilated_min) / (dilated_max - dilated_min + 1e-08)
        dilated_soft = torch.sigmoid((dilated_norm - 0.5) * sharpness)
        return dilated_soft * edge_map

    def adaptive_threshold(self, edge_map, k=0.2):
        flat = edge_map.view(-1)
        k_int = max(1, int(flat.size(0) * k))
        threshold = torch.topk(flat, k_int)[0][-1]
        return edge_map * (edge_map > threshold).float()

    def refine_edges_with_cnn(self, x, grad):
        """Fuse the image and gradient features with the edge-refinement CNN."""
        feat = self.backbone(torch.cat([x, grad], dim=1))
        return feat

    def edge_aware_filtering(self, image):
        """Apply the EF-ALBP edge-aware filter."""
        tau_clamped = torch.clamp(self.tau, min=0.001, max=1.0)
        return edge_aware_filtering_torch(image, sigma_s=self.sigma_s, sigma_r=self.sigma_r, tau=tau_clamped)

    def albp(self, filtered_img):
        """Compute the ALBP edge response."""
        return albp_torch(filtered_img, alphaT=self.alphaT)

    def forward(self, x, test=False, mask=None):
        assert self.alphaT.requires_grad, 'alphaT should require grad'
        B, C, H, W = x.shape
        hair_attenuation_mask = self.hair_removal_net(x)
        x_dhair = x * hair_attenuation_mask
        filtered_img = self.edge_aware_filtering(x_dhair)
        edge_map = self.albp(filtered_img)
        edge_map = self.multi_scale_filtering(edge_map)
        edge_map_dilated = self.morphological_processing(edge_map, kernel_size=3)
        em_mean = edge_map_dilated.mean(dim=[2, 3], keepdim=True)
        em_std = edge_map_dilated.std(dim=[2, 3], keepdim=True) + 1e-08
        edge_map_dilated = (edge_map_dilated - em_mean) / em_std
        edge_map_dilated = edge_map_dilated * self.edge_scale + self.edge_bias
        edge_map_dilated = self.soft_morphological_opening(edge_map_dilated, kernel_size=3)

        def _adaptive_threshold_safe(tensor, k=0.2):
            flat = tensor.view(-1)
            k_int = max(1, int(flat.numel() * k))
            threshold = torch.topk(flat, k_int)[0][-1]
            return tensor * (tensor > threshold).float()
        edge_map_dilated = _adaptive_threshold_safe(edge_map_dilated, k=0.2)
        grad_mag = self.compute_gradient(x_dhair)
        edge_map_dilated = self.soft_gradient_guided_refinement(edge_map_dilated, grad_mag)
        edge_map_dilated = torch.sigmoid(edge_map_dilated)
        em_min = edge_map_dilated.amin(dim=[2, 3], keepdim=True)
        em_max = edge_map_dilated.amax(dim=[2, 3], keepdim=True)
        edge_map_dilated = (edge_map_dilated - em_min) / (em_max - em_min + 1e-08)
        if self.debug and random.random() < 0.0007:
            self._debug_outputs(x, edge_map_dilated, None if mask is None else generate_medical_edges_gpu(mask, dilation=18))
        if test:
            return {'edge_map': edge_map_dilated}
        return edge_map_dilated

    def _debug_outputs(self, x, edge_map, gt_edge):

        def stats(tensor, name):
            t = tensor.detach().float().cpu()
            flat = t.view(-1).numpy()
            import numpy as np
            nan_cnt = np.isnan(flat).sum()
            inf_cnt = np.isinf(flat).sum()
            pct_pos = (flat > 0.5).sum() / max(1, flat.size)
            ptiles = np.percentile(flat, [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
            print(f'[{name}] shape={tuple(tensor.shape)} min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}, std={flat.std():.6f}, nan={nan_cnt}, inf={inf_cnt}, >0.5_ratio={pct_pos:.4f}')
            print(f'  {name} percentiles (0.1,1,5,10,25,50,75,90,95,99,99.9): {ptiles}')
        print(f'[Enhancer Debug] training={self.training}, alphaT={float(self.alphaT.item()):.4f}, tau={float(self.tau.item()):.4f}, sigma_s={float(self.sigma_s.item()):.4f}, sigma_r={float(self.sigma_r.item()):.4f}')
        stats(edge_map, 'edge_map')
        if gt_edge is not None:
            stats(gt_edge, 'gt_edge')
        if gt_edge is not None:
            iou = compute_iou(edge_map, gt_edge)
            print(f'[Debug] IoU with ground truth edge: {iou:.4f}')
            fg_mean = (edge_map * gt_edge).sum() / (gt_edge.sum() + 1e-08)
            bg_mean = (edge_map * (1 - gt_edge)).sum() / ((1 - gt_edge).sum() + 1e-08)
            edge_strength_ratio = fg_mean / (bg_mean + 1e-08)
            print(f'[Debug] Edge Strength Ratio: {edge_strength_ratio:.4f}')
            fg_active_ratio = (edge_map * gt_edge > 0.5).float().sum() / (gt_edge.sum() + 1e-08)
            print(f'[Debug] Foreground Active Ratio: {fg_active_ratio:.4f}')
        try:
            import os, numpy as np, matplotlib.pyplot as plt
            os.makedirs('enhancer_debug', exist_ok=True)
            import random
            step_str = f'step_{random.randint(1000, 9999)}'
            t_edge = edge_map.detach().float().cpu().numpy()[0, 0]
            if gt_edge is not None:
                t_gt = gt_edge.detach().float().cpu().numpy()[0, 0]
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(t_edge, cmap='gray', vmin=0, vmax=1)
            plt.title('Predicted Edge Map')
            plt.axis('off')
            if gt_edge is not None:
                plt.subplot(1, 3, 2)
                plt.imshow(t_gt, cmap='gray', vmin=0, vmax=1)
                plt.title('GT Edge Map')
                plt.axis('off')
            else:
                plt.subplot(1, 3, 2)
                plt.text(0.5, 0.5, 'No GT', horizontalalignment='center', verticalalignment='center')
                plt.title('GT Edge Map')
                plt.axis('off')
            if gt_edge is not None:
                overlay = np.zeros((t_edge.shape[0], t_edge.shape[1], 3))
                overlay[..., 0] = t_edge
                overlay[..., 1] = t_gt
                overlay = np.clip(overlay, 0, 1)
                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title('Overlay (Pred:Red, GT:Green)')
                plt.axis('off')
            else:
                plt.subplot(1, 3, 3)
                plt.text(0.5, 0.5, 'No GT', horizontalalignment='center', verticalalignment='center')
                plt.title('Overlay')
                plt.axis('off')
            plt.tight_layout()
            save_name = os.path.join('enhancer_debug', f'enhancer_debug_{step_str}.png')
            plt.savefig(save_name, dpi=180, bbox_inches='tight')
            plt.close()
            print(f'[Enhancer Debug] Saved visual to {save_name}')
        except Exception as e:
            print('[Enhancer Debug] Failed to save viz:', str(e))

def compute_iou(pred_edge, gt_edge, thresh=None):
    """Compute the mean edge IoU for a batch."""
    p = pred_edge.detach().cpu().numpy()
    g = gt_edge.detach().cpu().numpy()
    B = p.shape[0]
    ious = []
    for b in range(B):
        pb = p[b, 0]
        gb = g[b, 0]
        if thresh is None:
            t = pb.mean()
        else:
            t = float(thresh)
        pb_bin = (pb > t).astype(np.uint8)
        gb_bin = (gb > 0.5).astype(np.uint8)
        iou = compute_relaxed_iou(gb_bin[None, ...], pb_bin[None, ...], kernel_size=1)
        ious.append(iou)
    return float(np.mean(ious))

class EdgeEnhanceCNN(nn.Module):

    def __init__(self, in_channels=1):
        super(EdgeEnhanceCNN, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, x):
        return x + self.encoder(x)

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        """
        F_g: gating feature channel (decoder feature)
        F_l: skip connection / edge feature channel
        F_int: intermediate channel
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        gated = x * psi
        a = torch.clamp(self.alpha, 0.0, 1.0)
        out = g + a * gated
        return out
try:
    import kornia
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

def gaussian_kernel1d(kernel_size: int, sigma: float, device=None, dtype=torch.float32):
    half = (kernel_size - 1) / 2.0
    x = torch.linspace(-half, half, steps=kernel_size, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def separable_gaussian_blur(img: torch.Tensor, kernel_size: int=9, sigma: float=3.0):
    """
    img: tensor [B,1,H,W] or [B,C,H,W], dtype float, on device
    returns blurred image same shape
    Uses separable conv (horizontal then vertical) with group conv for speed.
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype
    k1 = gaussian_kernel1d(kernel_size, sigma, device=device, dtype=dtype)
    k_h = k1.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    k_v = k1.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
    padding_h = (0, (kernel_size - 1) // 2)
    padding_v = ((kernel_size - 1) // 2, 0)
    x = F.pad(img, (padding_h[0], padding_h[1], padding_h[0], padding_h[1]), mode='reflect') if False else F.pad(img, ((kernel_size - 1) // 2, (kernel_size - 1) // 2, 0, 0), mode='reflect')
    out = F.conv2d(F.pad(img, ((kernel_size - 1) // 2, (kernel_size - 1) // 2, 0, 0), mode='reflect'), weight=k_h, groups=C)
    out = F.conv2d(F.pad(out, (0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2), mode='reflect'), weight=k_v, groups=C)
    return out

def denoise_gray_on_gpu(gray: torch.Tensor, method: str='auto'):
    """
    gray: [B,1,H,W], dtype float on device
    method: "auto" tries kornia -> fallback to separable gaussian
    """
    if _HAS_KORNIA and method in ('auto', 'kornia'):
        try:
            return kornia.filters.bilateral_blur(gray, (9, 9), sigma_color=0.08, sigma_space=9.0)
        except Exception:
            pass
    return separable_gaussian_blur(gray, kernel_size=9, sigma=2.5)

class EdgeEnhancedUNet(nn.Module):
    """UNet with edge enhancement modules"""

    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        self.fixed_weights = {'seg': 1.0, 'edge': 0.5}
        _init_weights = [0.6, 0.1, 0.4, 0.1]
        init_log_vars = [-math.log(w) for w in _init_weights]
        self.loss_log_vars = nn.ParameterList([nn.Parameter(torch.tensor(v, dtype=torch.float32)) for v in init_log_vars])
        self.shallow_edge = ShallowEdgeExtractor()
        self.edge_fuse = nn.Sequential(nn.Conv2d(32 + 256, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.edge_module = EnhancedEdgeModule(in_channels=32, decoder_out_channels=128)
        self.ef_weight = nn.Parameter(torch.tensor(1.0))
        self.enhanced_weight = nn.Parameter(torch.tensor(1.0))
        self.fuse_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(128, 16, 1), nn.BatchNorm2d(16), nn.ReLU()), nn.Sequential(nn.Conv2d(128, 32, 1), nn.BatchNorm2d(32), nn.ReLU()), nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU()), nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())])
        self.shallow_edge_head = nn.ModuleList([nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1)), nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=1))])
        self.raw_shallow_scales = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32))
        self.fuse_post_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))])
        self.shallow_gate = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.shallow_gate.weight, 0.0)
        nn.init.constant_(self.shallow_gate.bias, 0.0)
        self.shallow_scales = nn.ParameterList([nn.Parameter(torch.tensor(0.5, dtype=torch.float32)) for _ in range(4)])
        self.ef_albp_cnn = LearnableEdgeEnhancer(in_channels=1, sigma_s=5, sigma_r=0.5, tau=0.05, alphaT=2.0, debug=True)
        self.enhanced = EdgeEnhanceCNN(in_channels=1)
        self.num_classes = getattr(self.unet, 'num_classes', 2)
        self.edge_proj = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.edge_alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.fuse_convs_shallow = nn.ModuleList([nn.Conv2d(32, 16, 1), nn.Conv2d(32, 32, 1), nn.Conv2d(32, 64, 1), nn.Conv2d(32, 128, 1)])
        self.att_gates = nn.ModuleList([AttentionGate(16, 16, 8), AttentionGate(32, 32, 16), AttentionGate(64, 64, 32), AttentionGate(128, 128, 64)])

    def forward(self, x, mask=None, shallow_only=False, epoch_num=None, global_step=None, total_steps=None, return_aux=True):
        """Forward pass"""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-08)
        gray_denoised = denoise_gray_on_gpu(gray, method='auto')
        gray_denoised = gray_denoised.to(device=gray.device, dtype=gray.dtype)
        if self.training:
            ef_out = self.ef_albp_cnn(gray_denoised, mask=mask)
        else:
            ef_out = self.ef_albp_cnn(gray_denoised)
        enhanced = self.enhanced(gray)
        enhanced = F.interpolate(enhanced, size=ef_out.shape[2:], mode='bilinear', align_corners=False)
        shallow_in = torch.cat([self.ef_weight * ef_out, self.enhanced_weight * enhanced], dim=1)
        shallow_input_image = F.interpolate(ef_out, size=x.shape[2:], mode='bicubic', align_corners=False)
        if hasattr(self.unet, 'encoder') and callable(getattr(self.unet.encoder, '__call__', None)):
            features = self.unet.encoder(x)
            if not (isinstance(features, (list, tuple)) and len(features) == 5):
                raise RuntimeError(f"Encoder returned unexpected features: type={type(features)}, len={(len(features) if hasattr(features, '__len__') else 'N/A')}")
            x1, x2, x3, x4, x5 = features
        else:
            x1 = self.unet.encoder.in_conv(x)
            x2 = self.unet.encoder.down1(x1)
            x3 = self.unet.encoder.down2(x2)
            x4 = self.unet.encoder.down3(x3)
            x5 = self.unet.encoder.down4(x4)
            features = [x1, x2, x3, x4, x5]
        shallow_feat = self.shallow_edge(ef_input=shallow_in, x1_feat=x1)
        shallow_edge_feat_list = [head(shallow_feat) for head in self.shallow_edge_head]
        if shallow_only:
            return (None, None, None, None, shallow_edge_feat_list, None)
        edge_feat_down = F.interpolate(shallow_feat, size=x5.shape[2:], mode='bicubic', align_corners=False)
        edge_input = self.edge_fuse(torch.cat([edge_feat_down, x5], dim=1))
        edge_pred_main, edge_pred_low, edge_features = self.edge_module(edge_input)
        self.edge_features = edge_features
        seg_output, gate = self.decoder_forward(features, edge_features, shallow_feat)
        if self.training or return_aux:
            return (seg_output, edge_pred_main, edge_pred_low, features, shallow_edge_feat_list, shallow_input_image, gate)
        else:
            return (seg_output, edge_pred_main)

    def decoder_forward(self, features, edge_features, shallow_feat=None, epoch_num=None, warmup_epochs=5):
        """
        Gated decoder forward (additive shallow injection, stabilized):
          - fused = fuse_conv(edge_resized) + shallow_scale * gate_prob * s_proj
          - after fusion, pass through a small conv+bn (self.fuse_post_convs[idx]) to normalize
        """
        x1, x2, x3, x4, x5 = features

        def _compute_gate_prob(shallow_resized):
            gate_logits = self.shallow_gate(shallow_resized)
            gate_prob = torch.sigmoid(gate_logits)
            return gate_prob

        def _shallow_proj_at_size(shallow_feat_tensor, target_size, idx):
            shallow_resized = F.interpolate(shallow_feat_tensor, size=target_size, mode='bilinear', align_corners=False)
            shallow_proj = self.fuse_convs_shallow[idx](shallow_resized)
            return (shallow_resized, shallow_proj)

        def _zero_gate_for(edge_resized):
            B = edge_resized.shape[0]
            return torch.zeros((B, 1, edge_resized.shape[2], edge_resized.shape[3]), device=edge_resized.device, dtype=edge_resized.dtype)

        def _get_shallow_scale(idx, max_scale=0.5):
            raw = self.raw_shallow_scales[idx]
            return torch.sigmoid(raw) * float(max_scale)
        d1 = self.unet.decoder.up1(x5, x4)
        edge_resized = F.interpolate(edge_features, size=d1.shape[2:], mode='bilinear', align_corners=False)
        if shallow_feat is not None:
            s_res_1, s_proj_1 = _shallow_proj_at_size(shallow_feat, edge_resized.shape[2:], 3)
            gate1 = _compute_gate_prob(s_res_1)
            shallow_scale_1 = _get_shallow_scale(3)
            fused1 = self.fuse_convs[3](edge_resized) + shallow_scale_1 * gate1 * s_proj_1
        else:
            gate1 = _zero_gate_for(edge_resized)
            fused1 = self.fuse_convs[3](edge_resized)
        fused1 = self.fuse_post_convs[3](fused1)
        d1 = self.att_gates[3](d1, fused1)
        d2 = self.unet.decoder.up2(d1, x3)
        edge_resized = F.interpolate(edge_features, size=d2.shape[2:], mode='bilinear', align_corners=False)
        if shallow_feat is not None:
            s_res_2, s_proj_2 = _shallow_proj_at_size(shallow_feat, edge_resized.shape[2:], 2)
            gate2 = _compute_gate_prob(s_res_2)
            shallow_scale_2 = _get_shallow_scale(2)
            fused2 = self.fuse_convs[2](edge_resized) + shallow_scale_2 * gate2 * s_proj_2
        else:
            gate2 = _zero_gate_for(edge_resized)
            fused2 = self.fuse_convs[2](edge_resized)
        fused2 = self.fuse_post_convs[2](fused2)
        d2 = self.att_gates[2](d2, fused2)
        d3 = self.unet.decoder.up3(d2, x2)
        edge_resized = F.interpolate(edge_features, size=d3.shape[2:], mode='bilinear', align_corners=False)
        if shallow_feat is not None:
            s_res_3, s_proj_3 = _shallow_proj_at_size(shallow_feat, edge_resized.shape[2:], 1)
            gate3 = _compute_gate_prob(s_res_3)
            shallow_scale_3 = _get_shallow_scale(1)
            fused3 = self.fuse_convs[1](edge_resized) + shallow_scale_3 * gate3 * s_proj_3
        else:
            gate3 = _zero_gate_for(edge_resized)
            fused3 = self.fuse_convs[1](edge_resized)
        fused3 = self.fuse_post_convs[1](fused3)
        d3 = self.att_gates[1](d3, fused3)
        d4 = self.unet.decoder.up4(d3, x1)
        edge_resized = F.interpolate(edge_features, size=d4.shape[2:], mode='bilinear', align_corners=False)
        if shallow_feat is not None:
            s_res_4, s_proj_4 = _shallow_proj_at_size(shallow_feat, edge_resized.shape[2:], 0)
            gate4 = _compute_gate_prob(s_res_4)
            shallow_scale_4 = _get_shallow_scale(0)
            fused4 = self.fuse_convs[0](edge_resized) + shallow_scale_4 * gate4 * s_proj_4
        else:
            gate4 = _zero_gate_for(edge_resized)
            fused4 = self.fuse_convs[0](edge_resized)
        fused4 = self.fuse_post_convs[0](fused4)
        d4 = self.att_gates[0](d4, fused4)
        out = self.unet.decoder.out_conv(d4)
        edge_resized_final = F.interpolate(edge_features, size=out.shape[2:], mode='bilinear', align_corners=False)
        edge_logits = self.edge_proj(edge_resized_final)
        if out.shape[1] != edge_logits.shape[1]:
            raise RuntimeError(f'Channel mismatch: out {out.shape[1]} vs edge {edge_logits.shape[1]}')
        out = out + torch.clamp(self.edge_alpha, 0.0, 1.0) * edge_logits
        gate_up = gate4
        if gate_up is not None:
            gate_up = F.interpolate(gate_up, size=out.shape[2:], mode='bilinear', align_corners=False)
        return (out, gate_up)

class DiceLoss(nn.Module):
    """Soft Dice loss used by the segmentation and edge objectives."""

    def __init__(self, smooth: float=1e-05):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(pred.device)
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        denominator = (pred + target).sum()
        return 1 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

def weighted_bce_loss(pred, target, pos_weight=3.0, neg_weight=0.5):
    """Compute weighted binary cross-entropy on edge logits."""
    weight = torch.ones_like(target)
    weight[target > 0.5] = pos_weight
    weight[target <= 0.5] = neg_weight
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return (loss * weight).mean()

class EdgeFocalLoss(nn.Module):
    """Focal loss variant for edge detection"""

    def __init__(self, gamma=2.0, pos_weight=2.0, neg_weight=0.5):
        super(EdgeFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, input_logits, target):
        """
        Forward pass
        Args:
            input_logits: Raw logits [B,1,H,W]
            target: Binary edge GT [B,1,H,W] (0 or 1)
        """
        prob = torch.sigmoid(input_logits)
        prob = torch.clamp(prob, min=1e-06, max=1 - 1e-06)
        pt = torch.where(target == 1, prob, 1 - prob)
        weight = torch.where(target == 1, self.pos_weight, self.neg_weight)
        loss = -weight * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()

def plot_edge_histogram(edge_pred_tensor, tag='edge_pred', save_path=None):
    """
    Plot histogram of edge predictions
    Args:
        edge_pred_tensor: Tensor [B,1,H,W] of sigmoid outputs
        tag: Plot title tag
        save_path: Path to save plot (None to display)
    """
    pred_np = edge_pred_tensor.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(pred_np, bins=50, range=(0, 1), color='skyblue')
    plt.title(f'{tag} value distribution')
    plt.xlabel('Prediction value')
    plt.ylabel('Pixel count')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compute_relaxed_iou(gt, pred, kernel_size=9):
    """
    Compute IoU with relaxed matching (dilated GT)
    Args:
        gt: Ground truth array
        pred: Prediction array
        kernel_size: Dilation kernel size
    Returns:
        Relaxed IoU score
    """
    gt = gt.squeeze().astype(np.uint8)
    pred = pred.squeeze().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gt_d = cv2.dilate(gt, kernel, iterations=1)
    inter = np.logical_and(gt_d, pred).sum()
    union = np.logical_or(gt_d, pred).sum()
    return inter / (union + 1e-06)

def dilate_edge_map(gt_edge, kernel_size=3, iterations=1):
    """
    Dilate edge map
    Args:
        gt_edge: Edge tensor [B,1,H,W]
        kernel_size: Dilation kernel size
        iterations: Number of dilations
    Returns:
        Dilated edge tensor
    """
    gt_np = gt_edge.cpu().numpy()
    dilated = []
    for edge in gt_np:
        edge_img = edge[0]
        edge_img = (edge_img * 255).astype(np.uint8)
        edge_dilated = cv2.dilate(edge_img, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
        edge_dilated = edge_dilated.astype(np.float32) / 255.0
        dilated.append(edge_dilated[None, ...])
    dilated = np.stack(dilated, axis=0)
    return torch.from_numpy(dilated).to(gt_edge.device)

def generate_distance_edge_heatmap(label_tensor, max_dist=10):
    """Generate a distance-weighted soft edge target."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if label_tensor.dim() == 3:
        label_tensor = label_tensor.unsqueeze(1)
    B, _, H, W = label_tensor.shape
    label_np = label_tensor.cpu().numpy().astype(np.uint8)
    heatmaps = []
    for b in range(B):
        mask = label_np[b, 0]
        if mask.max() == 0:
            heatmaps.append(torch.zeros((1, H, W), dtype=torch.float32))
            continue
        boundary = mask ^ binary_erosion(mask)
        dist_map = distance_transform_edt(mask)
        dist_map[boundary == 1] = 0
        dist_map = np.clip(dist_map, 0, max_dist)
        dist_map = dist_map / max_dist
        dist_map = 1.0 - dist_map
        heatmaps.append(torch.from_numpy(dist_map).unsqueeze(0))
    return torch.stack(heatmaps, dim=0).float().to(label_tensor.device)

def generate_medical_edges_gpu(mask, dilation=1):
    """Extract and optionally dilate mask boundaries on the GPU."""
    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = mask - eroded
    if dilation > 1:
        k_size = dilation if dilation % 2 == 1 else dilation + 1
        pad = k_size // 2
        edge = F.max_pool2d(edge, kernel_size=k_size, stride=1, padding=pad)
    return edge.clamp(0.0, 1.0)

def find_best_threshold(pred, gt, dilation=9):
    """
    Find optimal threshold for edge prediction
    Args:
        pred: Raw logits [1,1,H,W]
        gt: Ground truth [1,1,H,W]
        dilation: Dilation for relaxed matching
    Returns:
        best_thresh: Optimal threshold
        best_iou: Best IoU achieved
    """
    pred_sigmoid = torch.sigmoid(pred)[0, 0].cpu().numpy()
    gt_np = gt[0, 0].cpu().numpy().astype(np.uint8)
    gt_dilated = cv2.dilate(gt_np, np.ones((dilation, dilation), np.uint8), iterations=1)
    best_thresh = 0.0
    best_iou = 0.0
    for thresh in np.linspace(0.2, 0.6, 41):
        pred_bin = (pred_sigmoid > thresh).astype(np.uint8)
        inter = np.logical_and(pred_bin, gt_dilated).sum()
        union = np.logical_or(pred_bin, gt_dilated).sum()
        iou = inter / (union + 1e-06)
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
    return (best_thresh, best_iou)

def train(args, snapshot_path):
    """Main training function"""
    scaler = GradScaler()
    print(f'Initializing the Res50Mamba backbone...')
    res50_enc = Res50MambaEncoder(in_chans=3, ft_chns=[16, 32, 64, 128, 256], pretrained=True, pretrained_path=None)
    base_unet = UNet_Res50Mamba(in_chns=3, class_num=args.num_classes, ft_chns=[16, 32, 64, 128, 256], res50_encoder=res50_enc).cuda()
    model = EdgeEnhancedUNet(base_unet).cuda()
    print(f'Mamba-EdgeNet initialized with checkpoint-compatible module names.')
    torch.backends.cudnn.benchmark = True
    train_dataset = ISIC2016Dataset(args.root_path, split='train', transform=get_train_transform())
    val_dataset = ISIC2016Dataset(args.root_path, split='test', transform=get_val_transform())
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn, generator=g)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True, worker_init_fn=worker_init_fn, generator=g)
    base_lr = args.base_lr
    max_iterations = args.max_iterations
    iter_num = 0
    best_performance = 0.0
    best_dice = 0.0
    max_epoch = 200
    earlystop = 0
    start_epoch = 0
    patience_counter = 0
    iterations_per_epoch = len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    other_params = [p for n, p in model.named_parameters() if 'loss_log_vars' not in n]
    optimizer = optim.AdamW([{'params': other_params, 'lr': base_lr}, {'params': model.loss_log_vars, 'lr': base_lr * 0.1, 'weight_decay': 0.001}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - start_epoch, eta_min=1e-06)
    train_loss_list = []
    train_dice_list = []
    val_loss_list = []
    val_seg_loss_list = []
    val_dice_list = []
    lrs = []
    vis_dir = os.path.join(snapshot_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    epoch_num = 1
    max_alpha_t = 4.0
    warmup_epoch = 20
    total_steps = len(trainloader) * max_epoch
    global_step = 0
    patience_counter = 0
    for epoch_num in iterator:
        if epoch_num < start_epoch:
            continue
        epoch_train_loss = 0
        use_edge_dropout = epoch_num >= 40
        train_start_time = time.time()
        train_dice_total = 0.0
        gradient_norm = 1
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch = sampled_batch['image'].cuda().float()
            label_batch = sampled_batch['label'].cuda()
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                seg_output, edge_pred, edge_pred_low, features, shallow_edge_feat_list, shallow_input_image, gate = model(x=volume_batch, mask=label_batch, epoch_num=epoch_num, global_step=global_step, total_steps=total_steps)
                train_dice = DiceLoss()(seg_output[:, 1], label_batch)
                train_dice_total += 1 - train_dice.item()
                gt_edge = generate_medical_edges_gpu(label_batch, dilation=1)
                edge_pred = F.interpolate(edge_pred, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                edge_prob = torch.sigmoid(edge_pred)
                edge_pred_up = F.interpolate(edge_pred, size=gt_edge.shape[2:], mode='nearest')
                edge_pred_low_up = F.interpolate(edge_pred_low, size=gt_edge.shape[2:], mode='nearest')
                dilation_iter = 5
                dilated_gt_edge = dilate_edge_map(gt_edge, kernel_size=dilation_iter, iterations=1)
                edge_loss_fn = EdgeFocalLoss(gamma=2.0, pos_weight=3.0, neg_weight=0.5)
                soft_gt = generate_distance_edge_heatmap(label_batch.unsqueeze(1)).cuda()
                hard_edge = generate_medical_edges_gpu(label_batch).cuda()
                masked_soft_gt = soft_gt * hard_edge
                loss_edge = edge_loss_fn(edge_pred_up, dilated_gt_edge) + 0.3 * edge_loss_fn(edge_pred_low_up, dilated_gt_edge)
                train_loss_shallow_edge = 0.0
                shallow_dice = 0.0
                shallow_bce = 0.0
                for shallow_feat in shallow_edge_feat_list:
                    shallow_feat_up = F.interpolate(shallow_feat, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    bce_loss_fn = nn.BCEWithLogitsLoss()
                    dice_loss_fn = DiceLoss()
                    shallow_bce += weighted_bce_loss(shallow_feat_up, masked_soft_gt, pos_weight=3.0, neg_weight=0.3)
                    shallow_dice += dice_loss_fn(torch.sigmoid(shallow_feat_up), masked_soft_gt)
                shallow_loss_fn = EdgeFocalLoss(gamma=2.0, pos_weight=3.0, neg_weight=0.5)
                shallow_focal = 0.0
                for shallow_feat in shallow_edge_feat_list:
                    shallow_feat_up = F.interpolate(shallow_feat, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    shallow_focal += shallow_loss_fn(shallow_feat_up, masked_soft_gt)
                train_loss_shallow_edge = 0.3 * shallow_bce + 0.7 * shallow_focal
                dice_loss_fn = DiceLoss()
                if seg_output.dim() == 4 and seg_output.size(1) == 2:
                    ce_loss = F.cross_entropy(seg_output, label_batch.squeeze(1).long())
                    dice = dice_loss_fn(seg_output[:, 1], label_batch)
                    loss_seg = 0.5 * ce_loss + 0.5 * dice
                else:
                    seg_prob = torch.sigmoid(seg_output)
                    bce_loss = F.binary_cross_entropy(seg_prob, label_batch.unsqueeze(1).float())
                    dice = dice_loss_fn(seg_output, label_batch.unsqueeze(1).float())
                    loss_seg = 0.5 * bce_loss + 0.5 * dice
                with torch.no_grad():
                    seg_mask = torch.sigmoid(seg_output) > 0.5
                    seg_mask = seg_mask.float()
                    seg_edge = generate_medical_edges_gpu(seg_mask)
                shallow_edge_pred_up = F.interpolate(shallow_edge_feat_list[1], size=gt_edge.shape[2:], mode='bicubic', align_corners=False)
                shallow_edge_sigmoid = torch.sigmoid(shallow_edge_pred_up)
                loss_bi_consistency = F.mse_loss(torch.sigmoid(shallow_edge_pred_up), torch.sigmoid(edge_pred_up))
                cnn_input_img = shallow_input_image
                cnn_input_img_up = F.interpolate(cnn_input_img, size=gt_edge.shape[2:], mode='bicubic', align_corners=False)
                supervised_edge_gt = generate_distance_edge_heatmap(label_batch.unsqueeze(1)).cuda()
                cnn_bce = weighted_bce_loss(cnn_input_img_up, dilated_gt_edge, pos_weight=5.0, neg_weight=0.2)
                cnn_dice = DiceLoss()(cnn_input_img_up, dilated_gt_edge)
                loss_cnn_supervise = 0.7 * cnn_bce + 0.3 * cnn_dice
                sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=gate.device)
                sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=gate.device)
                sobel_x = sobel_x.to(dtype=gate.dtype)
                sobel_y = sobel_y.to(dtype=gate.dtype)
                gate_grad_x = F.conv2d(gate, sobel_x, padding=1)
                gate_grad_y = F.conv2d(gate, sobel_y, padding=1)
                gate_smoothness_loss = torch.mean(torch.abs(gate_grad_x)) + torch.mean(torch.abs(gate_grad_y))
                fixed_seg_w = float(model.fixed_weights['seg'])
                fixed_edge_w = float(model.fixed_weights['edge'])
                aux_losses = [train_loss_shallow_edge, loss_bi_consistency, loss_cnn_supervise, gate_smoothness_loss]
                w_min, w_max = (0.02, 0.5)
                s_min = -math.log(w_max)
                s_max = -math.log(w_min)
                with torch.no_grad():
                    for p in model.loss_log_vars:
                        p.data.clamp_(s_min, s_max)
                s_list = list(model.loss_log_vars)
                weights = [torch.exp(-s) for s in s_list]
                if epoch_num < warmup_epoch:
                    small_w = 0.05
                    weights_phase1 = [weights[0]] + [torch.tensor(small_w, device=device) for _ in weights[1:]]
                    total_loss = fixed_seg_w * loss_seg + fixed_edge_w * loss_edge
                    for wi, Li in zip(weights_phase1, aux_losses):
                        total_loss = total_loss + wi * Li
                else:
                    for p in model.loss_log_vars:
                        p.requires_grad = True
                    total_loss = fixed_seg_w * loss_seg + fixed_edge_w * loss_edge
                    for wi, Li, s in zip(weights, aux_losses, s_list):
                        total_loss = total_loss + wi * Li + 0.5 * s
                weighted_fixed = fixed_seg_w * loss_seg.detach()
                weighted_fixed += fixed_edge_w * loss_edge.detach()
                weighted_aux = 0.0
                if epoch_num >= warmup_epoch:
                    for wi, Li in zip(weights, aux_losses):
                        weighted_aux = weighted_aux + wi.detach() * Li.detach()
                weighted_only = weighted_fixed + weighted_aux
                weighted_only = weighted_fixed + weighted_aux
            global_step += 1
            scaler.scale(total_loss).backward()
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f'[WARNING] Loss is NaN/Inf (epoch {epoch_num}), skipping update')
                optimizer.zero_grad()
                continue
            if torch.isnan(seg_output).any() or torch.isinf(seg_output).any():
                print(f'[CRITICAL] seg_output contains NaN/Inf (epoch {epoch_num})')
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.unscale_(optimizer)
            if gradient_norm == 1:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'Gradient norm: {total_norm}')
            gradient_norm = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                try:
                    ef = model.ef_albp_cnn
                    ef.tau_scale.data.clamp_(0.01, 8.0)
                    ef.edge_scale.data.clamp_(0.01, 8.0)
                    ef.edge_amp.data.clamp_(0.1, 6.0)
                    ef.tau.data.clamp_(0.0001, 1.0)
                    ef.alpha_t.data.clamp_(0.1, 10.0)
                except Exception:
                    pass
            train_end_time = time.time()
            iter_num += 1
            epoch_train_loss += float(weighted_only.item())
            epoch_train_loss_with_reg = epoch_train_loss if 'epoch_train_loss_with_reg' not in locals() else epoch_train_loss_with_reg
        enh_map = shallow_input_image
        min_val = enh_map.min().item()
        max_val = enh_map.max().item()
        mean_val = enh_map.mean().item()
        if max_val - min_val < 0.001 or mean_val < 0.01:
            print(f'[Warning] Enhanced map collapse: min={min_val:.5f}, max={max_val:.5f}, mean={mean_val:.5f}')
        avg_train_dice = train_dice_total / len(trainloader)
        train_dice_list.append(avg_train_dice)
        lrs.append(optimizer.param_groups[0]['lr'])
        if epoch_num % 1 == 0 and epoch_num != 0:
            logging.info(f'[Train] Epoch {epoch_num:03d} | Batch {i_batch:03d} | train Time: {train_end_time - train_start_time:.3f}s | train_dice{avg_train_dice:.4f} ')
        if epoch_num % 5 == 0 and iter_num != 0:
            with torch.no_grad():
                gt_edge_np = gt_edge[0, 0].float()
                gt_edge = gt_edge.float()
                gt_bin = gt_edge.cpu().numpy().astype(np.float32)
                gt_bin = np.squeeze(gt_bin)
                edge_pred_up = F.interpolate(edge_pred, size=gt_edge.shape[2:], mode='bicubic', align_corners=False)
                edge_prob = torch.sigmoid(edge_pred_up)
                edge_pred_bin = (edge_prob > 0.45).float()
                gt_np = gt_edge[0, 0].detach().cpu().numpy()
                pred_np = edge_pred_bin[0, 0].detach().cpu().numpy()
                edge_pred_best_thresh, edge_iou = find_best_threshold(edge_pred_up, gt_edge)
                shallow_pred_best_thresh, shallow_iou = find_best_threshold(shallow_edge_pred_up, gt_edge)
                aux_losses = [train_loss_shallow_edge, loss_bi_consistency, loss_cnn_supervise, gate_smoothness_loss]
                w_min, w_max = (0.02, 0.5)
                s_min = -math.log(w_max)
                s_max = -math.log(w_min)
                with torch.no_grad():
                    for p in model.loss_log_vars:
                        p.data.clamp_(s_min, s_max)
                log_vars = torch.stack([p.data for p in model.loss_log_vars])
                weights = torch.exp(-log_vars)
                weights = torch.clamp(weights, min=0.0001, max=0.5)
                weights_np = weights.detach().cpu().numpy().tolist()
                logging.info(f'epoch {epoch_num} | weighted_only: {weighted_only:.4f} | total_with_reg: {total_loss.item():.4f} (Seg: {loss_seg.item():.4f}, Edge: {loss_edge.item():.4f}, shallow: {train_loss_shallow_edge.item():.4f}, w={weights_np[0]:.4f}; bi: {loss_bi_consistency.item():.4f}, w={weights_np[1]:.4f}; cnn: {loss_cnn_supervise.item():.4f}, w={weights_np[2]:.4f}; gate: {gate_smoothness_loss.item():.6f}, w={weights_np[3]:.4f})')
            logging.info(f'Edge IoU: {edge_iou:.4f},edge thresh: {edge_pred_best_thresh:.4f},shallow IOU:{shallow_iou:.4f} ,shallow thresh: {shallow_pred_best_thresh:.4f}| ')
            with torch.no_grad():
                if gate is None:
                    print('[Monitor] gate is None')
                else:
                    gp = gate
                    try:
                        gp_mean = float(gp.mean().item())
                        gp_med = float(gp.flatten().median().item())
                        gp_max = float(gp.max().item())
                        print(f'[Monitor] gate mean/median/max: {gp_mean:.6f}/{gp_med:.6f}/{gp_max:.6f}')
                    except Exception as e:
                        print('[Monitor] gate stats error:', e)
                try:
                    shallow_feat_tensor = shallow_feat
                    target_h, target_w = (features[0].shape[2], features[0].shape[3])
                    s_res_4 = F.interpolate(shallow_feat_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    s_proj_4 = model.fuse_convs_shallow[0](s_res_4)
                    abs_mean = float(s_proj_4.detach().abs().mean().item())
                    abs_max = float(s_proj_4.detach().abs().max().item())
                    print(f'[Monitor] s_proj_4 abs mean/max: {abs_mean:.6f}/{abs_max:.6f}')
                except Exception as e:
                    print('[Monitor] s_proj_4 recompute skipped (error):', e)
                try:
                    raw_scales = model.raw_shallow_scales.detach().cpu().numpy().tolist()
                    sigmoid_scales = [float((torch.sigmoid(p) * 0.5).item()) for p in model.raw_shallow_scales]
                    print(f'[Monitor] raw_shallow_scales: {raw_scales}')
                    print(f'[Monitor] sigmoid*0.5 scales: {sigmoid_scales}')
                except Exception as e:
                    print('[Monitor] raw_shallow_scales error:', e)
        train_loss_list.append(epoch_train_loss / len(trainloader))
        if epoch_num % 3 == 0 and iter_num != 0:
            model.eval()
            val_start_time = time.time()
            val_total_dice = 0
            val_total_loss = 0
            val_total_seg_loss = 0
            with torch.no_grad():
                for sampled_batch in valloader:
                    image = sampled_batch['image'].cuda()
                    label = sampled_batch['label'].cuda()
                    seg_output, edge_pred, edge_pred_low, features, shallow_edge_feat_list, shallow_input_image, gate = model(image, epoch_num=epoch_num, global_step=global_step, total_steps=total_steps)
                    gt_edge = generate_medical_edges_gpu(label)
                    edge_pred = F.interpolate(edge_pred, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    edge_pred_up = F.interpolate(edge_pred, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    edge_pred_low_up = F.interpolate(edge_pred_low, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    soft_gt = generate_distance_edge_heatmap(label.unsqueeze(1)).cuda()
                    hard_edge = generate_medical_edges_gpu(label).cuda()
                    masked_soft_gt = soft_gt * hard_edge
                    edge_focal_fn = EdgeFocalLoss(gamma=3.0, pos_weight=5.0, neg_weight=0.2)
                    dilated_gt_edge = generate_medical_edges_gpu(label, dilation=5)
                    val_loss_edge = edge_focal_fn(edge_pred_up, dilated_gt_edge)
                    val_loss_edge += 0.3 * edge_focal_fn(edge_pred_low_up, dilated_gt_edge)
                    val_loss_shallow_edge = 0.0
                    shallow_loss_fn = EdgeFocalLoss(gamma=2.0, pos_weight=3.0, neg_weight=0.5)
                    val_shallow_focal = 0.0
                    for shallow_feat in shallow_edge_feat_list:
                        shallow_feat_up = F.interpolate(shallow_feat, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                        val_shallow_focal += shallow_loss_fn(shallow_feat_up, masked_soft_gt)
                    val_shallow_dice = 0.0
                    val_shallow_bce = 0.0
                    for shallow_feat in shallow_edge_feat_list:
                        shallow_feat_up = F.interpolate(shallow_feat, size=gt_edge.shape[2:], mode='bicubic', align_corners=False)
                        bce_loss_fn = nn.BCEWithLogitsLoss()
                        dice_loss_fn = DiceLoss()
                        bce = bce_loss_fn(shallow_feat_up, masked_soft_gt)
                        val_shallow_bce += weighted_bce_loss(shallow_feat_up, masked_soft_gt, pos_weight=3.0, neg_weight=0.3)
                        val_shallow_dice += dice_loss_fn(torch.sigmoid(shallow_feat_up), masked_soft_gt)
                    val_loss_shallow_edge = 0.3 * val_shallow_bce + 0.7 * val_shallow_focal
                    shallow_edge_pred_up = F.interpolate(shallow_edge_feat_list[1], size=gt_edge.shape[2:], mode='bicubic', align_corners=False)
                    if seg_output.dim() == 4 and seg_output.size(1) == 2:
                        ce_loss = F.cross_entropy(seg_output, label.squeeze(1).long())
                        dice = dice_loss_fn(seg_output[:, 1], label)
                        val_loss_seg = 0.5 * ce_loss + 0.5 * dice
                    else:
                        seg_prob = torch.sigmoid(seg_output)
                        bce_loss = F.binary_cross_entropy(seg_prob, label.unsqueeze(1).float())
                        dice = DiceLoss()(seg_output, label.unsqueeze(1).float())
                        val_loss_seg = 0.5 * bce_loss + 0.5 * dice
                    seg_mask = (torch.sigmoid(seg_output) > 0.5).float()
                    seg_edge = generate_medical_edges_gpu(seg_mask)
                    if seg_edge.shape != gt_edge.shape:
                        if seg_edge.dim() == 5:
                            seg_edge = seg_edge[:, :, 0]
                        if seg_edge.dim() == 3:
                            seg_edge = seg_edge.unsqueeze(1)
                        seg_edge = F.interpolate(seg_edge, size=gt_edge.shape[2:], mode='bicubic', align_corners=False, antialias=False)
                    loss_bi_consistency = F.mse_loss(torch.sigmoid(shallow_edge_pred_up), torch.sigmoid(edge_pred_up))
                    val_total_seg_loss += loss_seg.item()
                    edge_pred_best_thresh, edge_iou = find_best_threshold(edge_pred_up, gt_edge)
                    shallow_pred_best_thresh, shallow_iou = find_best_threshold(shallow_edge_pred_up, gt_edge)
                    sobel_x = torch.tensor([[[[-1, 1]]]], dtype=torch.float32, device=gate.device)
                    sobel_y = torch.tensor([[[[-1], [1]]]], dtype=torch.float32, device=gate.device)
                    gate_grad_x = F.conv2d(gate, sobel_x, padding=0)
                    gate_grad_y = F.conv2d(gate, sobel_y, padding=0)
                    gate_smoothness_loss = torch.mean(torch.abs(gate_grad_x)) + torch.mean(torch.abs(gate_grad_y))
                    val_losses = [val_loss_seg, val_loss_edge, val_loss_shallow_edge, loss_bi_consistency, loss_cnn_supervise, gate_smoothness_loss]
                    main_w = [model.fixed_weights['seg'], model.fixed_weights['edge']]
                    s = torch.stack([p.detach() for p in model.loss_log_vars])
                    s = torch.clamp(s, min=s_min, max=s_max)
                    weights_val = torch.exp(-s)
                    weights_val = torch.exp(-s)
                    val_losses[3] = 0.0
                    val_losses[5] = 0.0
                    val_total_loss_item = 0.0
                    for w, l in zip(main_w, val_losses[:2]):
                        val_total_loss_item += w * l
                    for w, l in zip(weights_val, val_losses[2:]):
                        val_total_loss_item += w * l
                    val_total_loss += float(val_total_loss_item.item())
            val_end_time = time.time()
            logging.info(f'[Val] Epoch {epoch_num:03d} | val Time: {val_end_time - val_start_time:.3f}s')
            s = torch.stack([p.detach() for p in model.loss_log_vars])
            s = torch.clamp(s, min=s_min, max=s_max)
            val_weights = torch.exp(-s).cpu().numpy()
            logging.info(f'val_Loss: {val_total_loss / len(valloader):.4f} (Seg: {val_loss_seg:.4f}, Edge: {val_loss_edge:.4f}, shallow: {val_loss_shallow_edge:.4f}, loss_bi_consistency: {loss_bi_consistency:.4f}, loss_cnn_supervise: {loss_cnn_supervise:.4f}, gate_smoothness_loss: {gate_smoothness_loss:.4f}) | Weights: seg=1.0, edge=0.5, shallow={val_weights[0]:.3f}, bi={val_weights[1]:.3f}, cnn={val_weights[2]:.3f}, gate={val_weights[3]:.3f}')
            val_seg_loss_list.append(val_total_seg_loss / len(valloader))
            val_loss_list.append(val_total_loss / len(valloader))
        scheduler.step()
        if epoch_num % 10 == 0 and epoch_num != 0:
            idx = np.random.randint(0, len(valloader.dataset))
            sampled_batch = valloader.dataset[idx]
            image = sampled_batch['image'].unsqueeze(0).cuda()
            label_batch = sampled_batch['label'].unsqueeze(0).cuda()
            with torch.no_grad():
                seg_output, edge_pred, edge_pred_low, features, shallow_edge_feat_list, shallow_input_image, gate = model(image, epoch_num=epoch_num, global_step=global_step, total_steps=total_steps)
                edge_pred_up = F.interpolate(edge_pred, label_batch.unsqueeze(1).shape[2:], mode='bicubic', align_corners=False)
                edge_pred_prob = torch.sigmoid(edge_pred_up)
                edge_pred_bin = (edge_pred_prob > edge_pred_best_thresh).float()
                edge_prob = edge_pred_prob
                label_np = label_batch[0].detach().cpu().numpy()
                gt_bin = label_np.astype(np.uint8)
                gt_edge = find_boundaries(gt_bin, mode='inner').astype(np.uint8)
                gt_edge_diliated = cv2.dilate(gt_edge, np.ones((9, 9), np.uint8), iterations=1)
                pred_edge_np = (edge_prob[0, 0] > 0.45).float().cpu().numpy()
                shallow_edge_pred_up = F.interpolate(shallow_edge_feat_list[1], size=(label_batch.shape[-2], label_batch.shape[-1]), mode='bicubic', align_corners=False)
                shallow_edge_pred_up = torch.sigmoid(shallow_edge_pred_up)
                shallow_pred_np = shallow_edge_pred_up[0, 0].detach().cpu().numpy()
                shallow_bin = (shallow_pred_np > shallow_pred_best_thresh).astype(np.uint8)
                gt_edge_for_vis = generate_medical_edges_gpu(label_batch)[0, 0].cpu().numpy()
                gt_bin = (gt_edge_for_vis > 0.5).astype(np.uint8)
                gt_dilated = cv2.dilate(gt_bin, np.ones((3, 3), np.uint8), iterations=1)
                shallow_dilated = cv2.dilate(shallow_bin, np.ones((3, 3), np.uint8), iterations=1)
                tp = shallow_bin & gt_dilated
                fp = shallow_bin & ~gt_dilated
                fn = gt_bin & ~shallow_dilated
                overlay_relaxed = np.zeros((shallow_bin.shape[0], shallow_bin.shape[1], 3), dtype=np.uint8)
                overlay_relaxed[tp == 1] = [0, 255, 0]
                overlay_relaxed[fp == 1] = [255, 0, 0]
                overlay_relaxed[fn == 1] = [0, 0, 255]
                enhanced_img_np = shallow_input_image[0, 0].detach().cpu()
                cnn_prob = torch.sigmoid(enhanced_img_np)
            plot_shallow_pred_hist(torch.sigmoid(shallow_edge_feat_list[1]), epoch_num, save_dir='os.path.join(vis_dir, 'shallow_hist')')
            fig = plt.figure(figsize=(30, 6))
            ax1 = fig.add_subplot(1, 6, 1)
            ax1.imshow(label_np, cmap='gray')
            ax1.set_title('Label')
            ax1.axis('off')
            ax2 = fig.add_subplot(1, 6, 2)
            ax2.imshow(gt_edge_diliated, cmap='gray')
            ax2.set_title('GT Edge')
            ax2.axis('off')
            ax3 = fig.add_subplot(1, 6, 3)
            ax3.imshow(pred_edge_np, cmap='gray')
            ax3.set_title('Edge Pred')
            ax3.axis('off')
            ax4 = fig.add_subplot(1, 6, 4)
            ax4.imshow(enhanced_img_np, cmap='gray', vmin=0, vmax=1)
            ax4.set_title('CNN Enhanced Img')
            ax4.axis('off')
            ax5 = fig.add_subplot(1, 6, 5)
            ax5.imshow(gate[0, 0].cpu().numpy(), cmap='gray')
            ax5.set_title('shallow gate(heat map)')
            ax5.axis('off')
            ax6 = fig.add_subplot(1, 6, 6)
            ax6.imshow(overlay_relaxed)
            ax6.set_title('Shallow Pred vs GT (Relaxed)')
            ax6.axis('off')
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'edge_vis_iter_{iter_num}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            del fig
            save_path1 = os.path.join(vis_dir, f'edge_pred_prob_{iter_num}.png')
            plot_edge_histogram(edge_prob, tag='edge_prob', save_path=save_path1)
            logging.info(f'Saved edge visualization to {save_path}')
            del image, label_batch, seg_output, edge_pred, edge_pred_up, edge_pred_prob, edge_pred_bin
            del shallow_edge_feat_list, shallow_edge_pred_up, shallow_pred_np, shallow_bin
            del enhanced_img_np, cnn_prob, overlay_relaxed, gt_edge_diliated, pred_edge_np
            del gt_bin, gt_edge, gt_dilated, shallow_dilated, tp, fp, fn
            gc.collect()
            torch.cuda.empty_cache()
        if epoch_num % 1 == 0:
            model.eval()
            metric_list = []
            dice_scores = []
            iou_scores = []
            for i_batch, sampled_batch in enumerate(valloader):
                image = sampled_batch['image'].cuda()
                label = sampled_batch['label'].numpy()
                if image.size(1) == 1:
                    image = image.expand(-1, 3, -1, -1)
                with torch.no_grad():
                    model.eval()
                    for i in range(image.shape[0]):
                        img_i = image[i].unsqueeze(0)
                        label_i = label[i]
                        dice, iou, seg_output = test_single_volume(img_i.cpu(), label_i, model, classes=args.num_classes, total_steps=total_steps, global_step=global_step)
                        dice_scores.append(dice)
                        iou_scores.append(iou)
            if dice_scores and iou_scores:
                metric_array = np.column_stack((dice_scores, iou_scores))
                avg_metrics = np.mean(metric_array, axis=0)
                logging.info(f'Validation metrics - Dice: {avg_metrics[0]:.4f}, IoU: {avg_metrics[1]:.4f}')
                performance = avg_metrics[0]
            else:
                performance = 0.0
                logging.error('Validation metrics are incomplete.')
            val_dice_list.append(avg_metrics[0])
            logging.info(f'Current val_dice_list: {val_dice_list[-1]}')
            if performance > best_performance:
                best_performance = performance
                torch.save(model.state_dict(), os.path.join(snapshot_path, f'{args.model}_best_model.pth'))
            if val_dice_list[-1] > best_dice:
                best_dice = val_dice_list[-1]
                patience_counter = 0
            elif epoch_num > 110:
                patience_counter += 1
            if epoch_num == 139:
                patience_counter = 0
            if patience_counter > 15:
                print(f'Early stopping at epoch {epoch_num}')
                break
        if epoch_num % 30 == 0 and epoch_num != 0:
            save_path = os.path.join(snapshot_path, f'checkpoint_epoch_{epoch_num}.pth')
            save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'iter_num': iter_num, 'epoch_num': epoch_num, 'best_performance': best_performance, 'best_dice': best_dice, 'patience_counter': patience_counter}
            torch.save(save_dict, save_path)
            logging.info(f'Saved checkpoint at epoch {epoch_num}')
        if epoch_num >= 100 and epoch_num % 5 == 0:
            swa_snap_path = os.path.join(snapshot_path, f'swa_snap_epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), swa_snap_path)
            logging.info(f'====== [SWA Snapshot] Saved pure weight snapshot at epoch {epoch_num} ======')
        if epoch_num >= max_epoch:
            iterator.close()
            break
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    fig, ax1 = plt.subplots(figsize=(12, 8))
    epochs_train = range(1, len(train_loss_list) + 1)
    ax1.plot(epochs_train, train_loss_list, label='Train Loss', linewidth=2)
    ax1.plot(epochs_train, train_dice_list, label='Train Dice', linewidth=2)
    val_loss_epochs = list(range(3, len(train_loss_list) + 1, 3))
    val_loss_plot = val_loss_list[:len(val_loss_epochs)] if val_loss_list else []
    if len(val_loss_plot) > 0 and len(val_loss_epochs) == len(val_loss_plot):
        ax1.plot(val_loss_epochs, val_loss_plot, 'o-', label='Val Loss', markersize=6, color='red')
    val_dice_epochs = range(1, len(train_loss_list) + 1)
    val_dice_plot = val_dice_list[:len(val_dice_epochs)] if val_dice_list else []
    if len(val_dice_plot) > 0 and len(val_dice_epochs) == len(val_dice_plot):
        ax1.plot(val_dice_epochs, val_dice_plot, 's-', label='Val Dice', markersize=4, alpha=0.7, color='green')
    if val_dice_plot:
        best_val_dice = max(val_dice_plot)
        best_idx = val_dice_plot.index(best_val_dice)
        best_epoch = val_dice_epochs[best_idx]
        ax1.annotate(f'Best: {best_val_dice:.4f}', xy=(best_epoch, best_val_dice), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2 = ax1.twinx()
    if len(lrs) == len(epochs_train) and len(lrs) > 0:
        line_lr = ax2.plot(epochs_train, lrs, label='Learning Rate', color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Learning Rate', color='purple', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='purple')
    else:
        print("Warning: Length of 'lrs' does not match training epochs. Skipping LR plot.")
        line_lr = []
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss / Dice', fontsize=12)
    ax1.set_title('Training vs Validation Loss/Dice and Learning Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels() if len(lrs) == len(epochs_train) and len(lrs) > 0 else ([], [])
    all_lines = lines_1 + lines_2
    all_labels = labels_1 + labels_2
    if all_lines:
        ax1.legend(all_lines, all_labels, fontsize=10, loc='upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(snapshot_path, 'loss_dice_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info('Training Finished!')
    return 'Training Finished!'

def test_model(model, test_loader, device='cuda', save_dir=None):
    """Evaluate segmentation accuracy, boundary distance, and inference performance."""
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    failure_vis_dir = None
    if save_dir:
        failure_vis_dir = os.path.join(save_dir, 'failure_candidates')
        os.makedirs(failure_vis_dir, exist_ok=True)
    metrics = {k: [] for k in ['Dice', 'IOU', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'Recall', 'HD95', 'ASD', 'Edge_IoU']}
    is_cuda = torch.cuda.is_available() and 'cuda' in str(device)
    if is_cuda:
        print('Warming up GPU...')
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input, return_aux=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    total_time = 0.0
    total_samples = 0

    def compute_relaxed_iou(gt, pred, kernel_size=9):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gt_d = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1)
        pred_d = pred.astype(np.uint8)
        inter = np.logical_and(gt_d, pred_d).sum()
        union = np.logical_or(gt_d, pred_d).sum()
        return inter / (union + 1e-06)

    def generate_medical_edges(mask_tensor, dilation=2, device=device):
        if isinstance(mask_tensor, np.ndarray):
            m = torch.from_numpy(mask_tensor)
        else:
            m = mask_tensor
        if m.dim() == 3:
            m = m.unsqueeze(1)
        m_cpu = m.detach().cpu().numpy().astype(np.uint8)
        edges = []
        kernel = np.ones((dilation, dilation), np.uint8) if dilation > 0 else None
        for mask in m_cpu:
            arr = mask.squeeze(0) if mask.ndim == 3 else mask
            edge = find_boundaries(arr, mode='inner').astype(np.uint8)
            if kernel is not None:
                edge = cv2.dilate(edge, kernel, iterations=1)
            edges.append(edge[None, ...].astype(np.float32))
        edges_np = np.stack(edges, axis=0)
        return torch.from_numpy(edges_np).float().to(device)

    def _surface_distance(pred, target):
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return None
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        pred_surface = np.logical_xor(pred, binary_erosion(pred))
        target_surface = np.logical_xor(target, binary_erosion(target))
        dt_pred = distance_transform_edt(~pred_surface)
        dt_target = distance_transform_edt(~target_surface)
        pred_coords = np.argwhere(pred_surface)
        target_coords = np.argwhere(target_surface)
        if len(pred_coords) > 0:
            dist_pred_to_target = dt_target[pred_coords[:, 0], pred_coords[:, 1]]
        else:
            dist_pred_to_target = np.array([])
        if len(target_coords) > 0:
            dist_target_to_pred = dt_pred[target_coords[:, 0], target_coords[:, 1]]
        else:
            dist_target_to_pred = np.array([])
        all_surface_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
        return all_surface_distances if len(all_surface_distances) > 0 else None
    vis_dir = None
    if save_dir:
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing Mamba-EdgeNet')):
            images = batch['image'].to(device)
            labels_tensor = batch['label']
            labels_np = labels_tensor.detach().cpu().numpy()
            batch_size_current = images.size(0)
            if is_cuda:
                starter.record()
                outputs = model(images, return_aux=False)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000.0
            else:
                t0 = time.time()
                outputs = model(images, return_aux=False)
                curr_time = time.time() - t0
            total_time += curr_time
            total_samples += batch_size_current
            if isinstance(outputs, (tuple, list)):
                seg_output = outputs[0]
                edge_pred = outputs[1] if len(outputs) > 1 else None
            else:
                seg_output = outputs
                edge_pred = None
            seg_prob = torch.sigmoid(seg_output).detach().cpu().numpy()
            edge_prob_tensor = None
            if edge_pred is not None:
                edge_pred_cpu = edge_pred.cpu()
                target_size = (seg_prob.shape[-2], seg_prob.shape[-1])
                edge_pred_up = F.interpolate(edge_pred_cpu, size=target_size, mode='bilinear', align_corners=False)
                edge_prob_tensor = torch.sigmoid(edge_pred_up)
            for i in range(batch_size_current):
                if seg_prob.ndim == 4 and seg_prob.shape[1] == 2:
                    pred = seg_prob[i, 1]
                else:
                    pred = seg_prob[i, 0] if seg_prob.ndim == 4 else seg_prob[i]
                label = labels_np[i]
                if label.ndim == 3 and label.shape[0] == 1:
                    label = label[0]
                pred_bin = (pred > 0.5).astype(np.uint8)
                label_bin = (label > 0.5).astype(np.uint8)
                inter = np.sum(pred_bin * label_bin)
                dice = 2.0 * inter / (np.sum(pred_bin) + np.sum(label_bin) + 1e-07)
                union = np.sum(pred_bin) + np.sum(label_bin) - inter
                iou = inter / (union + 1e-07)
                if failure_vis_dir is not None and batch_idx % 10 == 0:
                    save_name = f'MambaEdgeNet_{batch_idx:04d}_Dice_{dice:.4f}.png'
                    save_path = os.path.join(failure_vis_dir, save_name)
                    pred_mask = pred_bin.astype(np.uint8) * 255
                    cv2.imwrite(save_path, pred_mask)
                tp = inter
                fp = np.sum(pred_bin * (1 - label_bin))
                fn = np.sum((1 - pred_bin) * label_bin)
                tn = np.sum((1 - pred_bin) * (1 - label_bin))
                acc = (tp + tn) / (tp + fp + fn + tn + 1e-07)
                prec = tp / (tp + fp + 1e-07)
                rec = tp / (tp + fn + 1e-07)
                sens = tp / (tp + fn + 1e-07)
                spec = tn / (tn + fp + 1e-07)
                try:
                    surface_dist = _surface_distance(pred_bin, label_bin)
                    if surface_dist is not None and len(surface_dist) > 0:
                        hd95 = np.percentile(surface_dist, 95)
                        asd = np.mean(surface_dist)
                    else:
                        hd95 = np.nan
                        asd = np.nan
                except Exception:
                    hd95 = np.nan
                    asd = np.nan
                metrics['Dice'].append(float(dice))
                metrics['IOU'].append(float(iou))
                metrics['HD95'].append(float(hd95) if not np.isnan(hd95) else np.nan)
                metrics['ASD'].append(float(asd) if not np.isnan(asd) else np.nan)
                metrics['Accuracy'].append(float(acc))
                metrics['Sensitivity'].append(float(sens))
                metrics['Specificity'].append(float(spec))
                metrics['Precision'].append(float(prec))
                metrics['Recall'].append(float(rec))

    def safe_mean(arr):
        arr = [v for v in arr if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        return np.mean(arr) if arr else np.nan
    avg = {k: safe_mean(v) for k, v in metrics.items()}
    std = {k: np.nanstd(v) if len(v) > 0 else np.nan for k, v in metrics.items()}
    fps = total_samples / total_time if total_time > 0 else 0.0
    avg_latency_ms = total_time / total_samples * 1000.0 if total_samples > 0 else 0.0
    max_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda else 0.0
    print('\n' + '=' * 25 + ' Hardware Performance ' + '=' * 25)
    print(f'Inference Latency  : {avg_latency_ms:.2f} ms / image')
    print(f'Inference FPS      : {fps:.2f} img/s')
    print(f'Max GPU Memory     : {max_memory_mb:.2f} MB')
    print('=' * 64)
    print('\nTest metrics (mean ± standard deviation):')
    for k in metrics:
        if k in avg and (not np.isnan(avg[k])):
            print(f'{k:<15}: {avg[k]:.4f} ± {std[k]:.4f}')
    print('=' * 64)
    with open('results.txt', 'a', encoding='utf-8') as f:
        content = '\n' + '=' * 30 + ' Test Results ' + '=' * 30 + '\n'
        content += '-' * 74 + '\n'
        content += f"{'Metric':<15}: {'Value':<10}\n"
        content += f"{'Dice':<15}: {avg['Dice']:.4f}\n"
        content += f"{'IoU':<15}: {avg['IOU']:.4f}\n"
        content += f"{'Accuracy':<15}: {avg['Accuracy']:.4f}\n"
        content += f"{'Sensitivity':<15}: {avg['Sensitivity']:.4f}\n"
        content += f"{'Specificity':<15}: {avg['Specificity']:.4f}\n"
        content += f"{'HD95':<15}: {avg['HD95']:.4f}\n"
        content += f"{'ASD':<15}: {avg['ASD']:.4f}\n"
        content += '=' * 74 + '\n'
        f.write(content)
    return avg
if __name__ == '__main__':
    for i in range(1):
        set_seed(i + 2036)
        with open('results.txt', 'a', encoding='utf-8') as f:
            content = f'Random seed: {i + 2036:.2f}\n'
            f.write(content)
            print(content)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        snapshot_path = os.path.join(args.output_dir, args.exp, args.model)
        os.makedirs(snapshot_path, exist_ok=True)
        logging.basicConfig(filename=snapshot_path + '/log.txt', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        if args.mode == 'train':
            train(args, snapshot_path)
        logging.info('\n' + '=' * 60)
        logging.info('Starting testing...')
        print(f'Initializing the Res50Mamba backbone...')
        res50_enc = Res50MambaEncoder(in_chans=3, ft_chns=[16, 32, 64, 128, 256], pretrained=True, pretrained_path=None)
        base_unet = UNet_Res50Mamba(in_chns=3, class_num=args.num_classes, ft_chns=[16, 32, 64, 128, 256], res50_encoder=res50_enc).cuda()
        model = EdgeEnhancedUNet(base_unet).cuda()
        print(f'Mamba-EdgeNet initialized with checkpoint-compatible module names.')
        best_model_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
        if not os.path.exists(best_model_path):
            logging.error(f'Best model not found at {best_model_path}. Skipping test.')
        else:
            state_dict = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            test_dataset = ISIC2016Dataset(args.root_path, split='test', transform=get_val_transform())
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=device.type == 'cuda')
            test_metrics = test_model(model, test_loader, device=device, save_dir=snapshot_path)
            with open(os.path.join(snapshot_path, 'test_results.txt'), 'w') as f:
                for k, metric_value in test_metrics.items():
                    try:
                        f.write(f'{k}: {metric_value:.4f}\n')
                        logging.info(f'{k}: {metric_value:.4f}')
                    except Exception:
                        f.write(f'{k}: {metric_value}\n')
                        logging.info(f'{k}: {metric_value}')
