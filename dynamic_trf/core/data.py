import torch

from tour.dataclass.stim import dictTensor_to

from . import TensorList, DictTensorList

CONTROL_STIM_TAG = 'control'
TARGET_STIM_TAG = 'target'
MODULATION_STIM_TAG = 'modulation'

class TorchDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        control_stims:TensorList, 
        target_stims:TensorList,
        modulation_stims:TensorList, 
        resps:TensorList, 
        device = 'cpu'
    ):  
        if len(control_stims) > 0:
            assert len(control_stims) == len(modulation_stims)
            assert len(control_stims) == len(resps)
        else:
            assert len(modulation_stims) == len(resps)
        self.control_stims = control_stims
        self.target_stims = target_stims
        self.modulation_stims = modulation_stims
        self.resps = resps
        self.device = device

    def __getitem__(self, index):
        # print('torchdata', index)
        stim_dict_tensor = {}
        if len(self.control_stims) > 0:
            stim_dict_tensor[CONTROL_STIM_TAG] = self.control_stims[index].to(self.device)
        stim_dict_tensor.update({
            TARGET_STIM_TAG: dictTensor_to(self.target_stims[index], self.device),
            MODULATION_STIM_TAG: dictTensor_to(self.modulation_stims[index], self.device)
        })
        resp = self.resps[index].to(self.device)
        return stim_dict_tensor, resp
    
    def __len__(self):
        return len(self.resps)