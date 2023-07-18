import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import Iterable, List


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class modelDataset(Dataset):
    """

        Custom pytorch data loader used for training. This class loads either 2D or
        3D models.

        Example useage for 3D models: dataset = modelDataset(directory='./models')
        
        Example usage for 2D models: dataset = modelDataset(directory='./models', dims=2)

    """

    def __init__(self, directory=None, mesh_dims=[10, 10], device='cpu'):
        
        self.file_list = []
        self.counter = 0
        self.device = device
        self.directory = None
        self.mesh_dims = mesh_dims

        # check if directory is empty
        try:
            
            self.check_directory_not_empty(directory)

            self.directory = directory
        
        except ValueError as e:
           
            print(str(e))

        # check if the directory and only use *.npy arrays
        try:

            self.file_list = self.filter_by_extension(os.listdir(directory), '.npy')

            # now create list of true models or labels
            train_list = []
            for file_id in self.file_list:

                if 'true' in file_id:

                    pass
                else:

                    train_list.append(file_id)

            self.file_list = train_list

        except ValueError as e:

            print(str(e))


    def __len__(self):
        
        return len(self.file_list)

    
    def __getitem__(self, index):
        
        filename = os.path.join(self.directory, self.file_list[index])

        numpy_array = np.load(filename)

        numpy_array = np.flip(numpy_array.reshape(59, 119), [1])

        dims = numpy_array.shape

        # padd for equal dims
        x1 = numpy_array.copy()
        x2 = np.zeros((dims[0], 4))
        x3 = np.zeros((dims[0], 5))
        nb = np.c_[x2, x1, x3]
        y1 = np.zeros((int(128 - dims[0]), 128))
        numpy_array = np.r_[nb, y1]

        tensor = torch.from_numpy(numpy_array).float().to(self.device)

        # now get the training
        file_name_label = filename.split('.')[0] + '_true.npy'
        numpy_array_label = np.load(file_name_label)
        numpy_array_label = np.flip(numpy_array_label.reshape(59, 119), [1])

        # padd for equal dims
        x1 = numpy_array_label.copy()
        x2 = np.zeros((dims[0], 4))
        x3 = np.zeros((dims[0], 5))
        nb = np.c_[x2, x1, x3]
        y1 = np.zeros((int(128 - dims[0]), 128))
        numpy_array_label = np.r_[nb, y1]

        tensor_label = torch.from_numpy(numpy_array_label).float().to(self.device)
        
        return tensor, tensor_label
    
    
    def filter_by_extension(self, string_list, extension):
        """

            Method that removes non-useful files defined by extension.

            :param string_list: list of files from a defined directory
            :type string_list: list
            :param extension: file extention to use
            :type extension: str

            :return: a list of the filtered file contents
            :rtype: list 
        
        """

        files_2_use = [s for s in string_list if s.endswith(extension)]

        if len(files_2_use) == 0:

            raise ValueError(f"no {extension} files found, please specify a correct directory")
        
        return files_2_use


    def check_directory_not_empty(self, path):
        """

            Checks to see if a directory is empty in which case no models can be loaded

            :param path: path to direcoty containing the training models.
            :type path: str
        
        """
        
        if not os.path.isdir(path):
            
            raise ValueError("Path is not a directory")

        if len(os.listdir(path)) == 0:
            
            raise ValueError("Directory is empty")


class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out
    

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk
        
class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x
    

class SegFormer(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation