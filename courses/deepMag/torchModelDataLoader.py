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

    def __init__(self, directory=None, dims='3', device='cpu', scale=1, scale_mean=1):
        
        self.file_list = []
        self.slice = 64
        self.counter = 0
        self.dims = dims
        self.count_2d = len(self.file_list)
        self.device = device
        self.directory = None
        self.scale = 1 / scale
        self.mean = scale_mean

        # check if directory is empty
        try:
            
            self.check_directory_not_empty(directory)

            self.directory = directory
        
        except ValueError as e:
           
            print(str(e))

        # check if the directory and only use *.npy arrays
        try:

            self.file_list = self.filter_by_extension(os.listdir(directory), '.npy')

        except ValueError as e:

            print(str(e))


    def __len__(self):
        
        return len(self.file_list)

    
    def __getitem__(self, index):
        
        filename = os.path.join(self.directory, self.file_list[index])
        numpy_array = np.load(filename) * self.scale
        tensor = torch.from_numpy(numpy_array).float().to(self.device)

        if self.dims == 2:
            
            self.counter += 1

            if self.counter == self.count_2d:
                self.slice = self.slice + 1
            
            return tensor[36:164, self.slice, :64]
        
        return tensor[36:164, 36:164, :64]
    
    
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


# --------------------------------------------------------------------------------------------

# testing the code

#

# import matplotlib.pyplot as plt

# batch_size = 32
# dataset = modelDataset(directory='./models', dims=3, scale=4.11)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# max_hold = []
# for i, data in enumerate(dataloader, 0):
    
#     print(data.shape, data.max())
#     max_hold.append(data.max())
#     # plt.imshow(data[0, :, :].cpu().detach().numpy().T)
#     # plt.show()

# print(f"max_data: {np.max(max_hold)}")