import torch
from tour.dataclass.dataset import Dataset
from tour.dataclass.io import stim_dict_from_hdf5
from tour.dataclass.stim import combine_stim_dict
from dynamic_trf.core import NestedTensorList, NestedTensorDictList
import argparse
from dynamic_trf.utils.io import (
    tour_stimdict_ndarray_to_tensor, tour_record_ndarray_to_tensor, cat_stim_by_feat_dim)

modulation_stim_names = ['lexical_surprisal', 'uniqueness_point']#, 'lexical_entropy']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./', help='Root directory for data files')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    data_root = args.data_root
    eeg_file = f"{data_root}/ns.h5"
    stim_file = f"{data_root}/ns_unipnt_lexsur_env_onset.h5"
    stim_file2 = f"{data_root}/oldman_lexical_entropy.h5"

    control_stims_name = ['envelope_fs64', 'word_onset_fs64']
    control_stims_combined_name = '+'.join(control_stims_name)
    target_stim_name = 'lexical_surprisal'
    modulation_stims_combined_name = '+'.join(modulation_stim_names)

    dataset = Dataset.load(eeg_file)
    stimuli_dict = stim_dict_from_hdf5(stim_file)
    stimuli_dict2 = stim_dict_from_hdf5(stim_file2)
    stimuli_dict = combine_stim_dict(stimuli_dict, stimuli_dict2)
    tour_stimdict_ndarray_to_tensor(stimuli_dict)
    tour_record_ndarray_to_tensor(dataset)
    cat_stim_by_feat_dim(stimuli_dict, control_stims_name, is_stimdict=False)
    cat_stim_by_feat_dim(stimuli_dict, modulation_stim_names, is_stimdict=True)
    dataset.stimuli_dict = stimuli_dict

    control_stims: NestedTensorList = []
    target_stims:  NestedTensorDictList = []
    modulation_stims:  NestedTensorDictList = []
    resps: NestedTensorList = []

    # iterate each subject
    for t_stims, t_resps, t_infos, t_k in dataset.to_pairs_iter():
        # print(t_k)
        # iterate each trial
        trial_control_stims, trial_target_stims, trial_modulation_stims, trial_resps = [], [], [], []
        for stim, t_resp in zip(t_stims, t_resps):
            t_control_stim = stim[control_stims_combined_name]
            t_target_stim = stim[target_stim_name]
            t_modulation_stim = stim[modulation_stims_combined_name]
            # if 'tag' in t_target_stim:
            #     del t_target_stim['tag']
            # if 'tag' in modulation_stims_combined_name:
            #     del t_modulation_stim['tag']
            assert torch.equal(t_target_stim['timeinfo'], t_modulation_stim['timeinfo'])

            target_len = torch.ceil(dataset.srate * t_target_stim['timeinfo'][1][-1]).long().numpy()
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

    torch.save(
        {
            'control_stims':control_stims,
            'target_stims': target_stims,
            'modulation_stims': modulation_stims,
            'resps': resps
        },
        f = f"{data_root}/dynamic_trf_input_examples.pt"
    )
