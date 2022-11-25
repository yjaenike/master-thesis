from data_utils.ts_dataset import TSDataset
from torch.utils.data import DataLoader

def prepare_dataloaders(data, batch_size=1, window_size=1,target_cols=None, pin_memory=True):
    '''
    
    Prepares a torch dataset and an iterable over the datset (dataloader) of a given numpy.ndarray dataset.
    
    :param data: dataset of type numpy.ndarray
    :param batch_size: the batch_size of the datalaoder obejct, defines how many samples are returned in each batch
    :param window_size: defines the sice of the sliding window that is applied to the timeseires 
    :param pin_memory: If ``True``, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
    :return dataset: An iterable Dataset.
    :return dataloader:  torch DataLoader obejct represents a python iterable over a dataset
    
    '''
    # create a Time Series Dataset object
    dataset = TSDataset(data, window_size, target_cols)
    
    # create a DataLoader object from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
    
    return dataset, dataloader