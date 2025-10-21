import os
import sys
import datetime
import warnings
from typing import List

import torch
import numpy as np
from tour.dataclass.dataset import Dataset, align_data




def checkFolder(folderPath):
#    print(folderPath)
#    if not isinstance(folderPath,str):
#        return
    if not os.path.isdir(folderPath) and not os.path.isfile(folderPath):
        warnings.warn("path: " + folderPath + " doesn't exist, and it is created")
        os.makedirs(folderPath)

''' Python Object IO'''
def pickle_load(filePath):
    import pickle
    file = open(filePath, 'rb')
    temp = pickle.load(file)
    return temp

def pickle_save(Object,folderName,tag=None, ext = '.bin'):
    if tag is None:
        file = open(folderName, 'wb')
    else:
        checkFolder(folderName)
        file = open(folderName + '/' + str(tag) + ext, 'wb')
    import pickle
    pickle.dump(Object,file)
    file.close()

def getUpperDir(path:str):
    return os.path.split(path)

def tour_stimdict_ndarray_to_tensor(stimuli_dict:dict):
    torch_default = torch.get_default_dtype()
    for stim_id, stim_feats in stimuli_dict.items():
        for feat_k, feat in stim_feats.items():
            if isinstance(feat, np.ndarray):
                stim_feats[feat_k] = torch.from_numpy(feat).to(torch_default)
            else:
                stim_feats[feat_k]['x'] = torch.from_numpy(feat['x']).to(torch_default)
                stim_feats[feat_k]['timeinfo'] = torch.from_numpy(feat['timeinfo']).to(torch_default)

def tour_record_ndarray_to_tensor(dataset:Dataset):
    torch_default = torch.get_default_dtype()
    for record in dataset.records:
        record.data = torch.from_numpy(record.data).to(torch_default)

def cat_stim_by_feat_dim(stimuli_dict:dict, feat_name: list, is_stimdict:bool = False):
    for stim_id, stim_feats in stimuli_dict.items():
        sel_stim_feats = [stim_feats[feat] for feat in feat_name]
        if not is_stimdict:
            sel_stim_feats = align_data(*sel_stim_feats)
            sel_stim_feats = torch.cat(sel_stim_feats, dim = 0)
            stim_feats['+'.join(feat_name)] = sel_stim_feats
        else:
            for sel_stim in sel_stim_feats:
                assert torch.equal(sel_stim['timeinfo'], sel_stim_feats[0]['timeinfo'])
                # assert sel_stim['tag'] == sel_stim_feats[0]['tag']
            new_x = torch.cat(
                [sel_stim['x'] for sel_stim in sel_stim_feats], dim = 0
            )
            stim_feats['+'.join(feat_name)] = {
                'x':  new_x,
                'timeinfo': sel_stim_feats[0]['timeinfo'],
                'tag': sel_stim_feats[0]['tag']
            }

def arrays_to_device(arrays:List[torch.Tensor], device) -> List[torch.Tensor]:
    return [arr.to(device) for arr in arrays]
