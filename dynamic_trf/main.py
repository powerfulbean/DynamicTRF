import os

import torch
import numpy as np

from tour.dataclass.dataset import Dataset
from tour.dataclass.io import stim_dict_from_hdf5

from dynamic_trf.utils.io import (
    tour_stimdict_ndarray_to_tensor, tour_record_ndarray_to_tensor, cat_stim_by_feat_dim)
from dynamic_trf.utils.args import get_arg_parser
from dynamic_trf.core import NestedArrayList, NestedArrayDictList, Configuration, engine

if __name__ == '__main__':


    ### prepare the dataset

    # load the paired stimuli and the responses

    # the stimuli contains continuous stimuli and discrete stimuli

    """
    the control_stims, target_stims, resps should be nested List of numpy array or StimDictArray (target_stims only)
        each item of the outer list corresponding to one subject, each item of the inner list corresponding to one trial
        the size of it is [# of subject * [# of trials * (n_samples, n_channels)]]
    """
    torch.set_default_dtype(torch.float64)
    data_root = r"F:"
    eeg_file = f"{data_root}/ns.h5"
    stim_file = f"{data_root}/ns_unipnt_lexsur_env_onset.h5"
    control_stims_name = ['envelope_fs64', 'word_onset_fs64']
    control_stims_combined_name = '+'.join(control_stims_name)
    target_stim_name = 'lexical_surprisal'
    modulation_stims_name = ['lexical_surprisal']
    modulation_stims_combined_name = '+'.join(modulation_stims_name)

    dataset = Dataset.load(eeg_file)
    stimuli_dict = stim_dict_from_hdf5(stim_file)
    tour_stimdict_ndarray_to_tensor(stimuli_dict)
    tour_record_ndarray_to_tensor(dataset)
    cat_stim_by_feat_dim(stimuli_dict, control_stims_name, is_stimdict=False)
    cat_stim_by_feat_dim(stimuli_dict, modulation_stims_name, is_stimdict=True)
    dataset.stimuli_dict = stimuli_dict

    control_stims: NestedArrayList = []
    target_stims:  NestedArrayDictList = []
    modulation_stims:  NestedArrayDictList = []
    resps: NestedArrayList = []

    # iterate each subject
    for t_stims, t_resps, t_infos, t_k in dataset.to_pairs_iter():
        # print(t_k)
        # iterate each trial
        trial_control_stims, trial_target_stims, trial_modulation_stims, trial_resps = [], [], [], []
        for stim, t_resp in zip(t_stims, t_resps):
            t_control_stim = stim[control_stims_combined_name]
            t_target_stim = stim[target_stim_name]
            t_modulation_stim = stim[modulation_stims_combined_name]
            if 'tag' in t_target_stim:
                del t_target_stim['tag']
            if 'tag' in modulation_stims_combined_name:
                del t_modulation_stim['tag']
            assert torch.equal(t_target_stim['timeinfo'], t_modulation_stim['timeinfo'])

            target_len = torch.round(dataset.srate * t_target_stim['timeinfo'][1][-1]).long().numpy()
            control_len = t_control_stim.shape[-1]
            resp_len = t_resp.shape[-1]
            assert control_len >= target_len and resp_len >= target_len, (target_len, control_len, resp_len)

            trial_control_stims.append(t_control_stim[:, :target_len])
            trial_target_stims.append(t_target_stim)
            trial_modulation_stims.append(t_modulation_stim)
            trial_resps.append(t_resp[:, :target_len])
        
        control_stims.append(trial_control_stims)
        target_stims.append(trial_target_stims)
        modulation_stims.append(trial_modulation_stims)
        resps.append(trial_resps)
    
    
    extraTimeLag = 200
    args = get_arg_parser()
    default_configs = vars(args).copy()
    user_configs = {
        'contextModel': 'CausalConv',
        'fTRFMode': '+-a,b', #real value amplitude scaling (a) amd time shifit (b)
        'fs': 64,
        'tarDirRoot': "F:",
        'extraTimeLag': 200,
        'device': 'cuda',
        'lr': (0.001, 0.01),
        'checkpoint': True
    }
    
    configs = default_configs.copy()
    configs.update(user_configs)

    configs = Configuration(**configs)

    assert configs.fs > 0

    engine.run(control_stims, target_stims, modulation_stims, resps, configs)