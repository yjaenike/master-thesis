import torch
from torch.utils.data import Dataset

class TSDataset(Dataset):
    def __init__(self, data, window, target_cols=None):
        '''
        :param data: dataset of type numpy.ndarray
        :param window: the windpow size of the time series sequence
        :param target_cols: specifies the target cols 
        :param shape:
        :param siez:
        '''
        self.data = torch.Tensor(data)
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()
        
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        
        # If the target cols are not specified, use all columns as targets
        if self.target_cols is None:
            self.target_cols = self.data.shape[1]
        y = torch.unsqueeze(self.data[index+self.window, -self.target_cols:], dim=0)
        return x, y
    
    def __len__(self):
        return len(self.data) - self.window
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())
        
    