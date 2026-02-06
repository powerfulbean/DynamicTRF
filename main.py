import os

import torch
import numpy as np
from enum import Enum

from dynamic_trf.utils.args import get_arg_parser
from dynamic_trf.core import Configuration, engine, NestedTensorDictList, StimDictTensor
from example_tourdataset import modulation_stim_names

class ExampleMode(Enum):
    control_modulation_lexsur = 'ctrl-mod~lexsur'
    modulation_lexsur = 'mod~lexsur'
    control_modulation_lexsur_unipnt = 'ctrl-mod~lexsur+unipnt'
    modulation_lexsur_unipnt = 'mod~lexsur+unipnt'
    control_modulation_unipnt = 'ctrl-mod~unipnt'
    modulation_unipnt = 'mod~unipnt'

class Config:

    include_control = True
    modulation_stim_names = modulation_stim_names



if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    mode = ExampleMode.modulation_unipnt
    if mode == ExampleMode.control_modulation_lexsur:
        Config.include_control = True
        Config.modulation_stim_names = [modulation_stim_names[0]]
    elif mode == ExampleMode.control_modulation_lexsur_unipnt:
        Config.include_control = True
        Config.modulation_stim_names = modulation_stim_names
    elif mode == ExampleMode.control_modulation_unipnt:
        Config.include_control = True
        Config.modulation_stim_names = [modulation_stim_names[1]]
    elif mode == ExampleMode.modulation_lexsur:
        Config.include_control = False
        Config.modulation_stim_names = [modulation_stim_names[0]]
    elif mode == ExampleMode.modulation_lexsur_unipnt:
        Config.include_control = False
        Config.modulation_stim_names = modulation_stim_names
    elif mode == ExampleMode.modulation_unipnt:
        Config.include_control = False
        Config.modulation_stim_names = [modulation_stim_names[1]]
    else:
        raise ValueError(f'{mode} not supproted')

    # load the paired stimuli and the responses
    # the stimuli contains continuous stimuli and discrete stimuli

    """
    the control_stims, target_stims, resps should be nested List of torch.Tensors or StimDictTensor (target_stims and modulation_stims)
        each item of the outer list corresponding to one subject, each item of the inner list corresponding to one trial
        the size of it is [# of subject * [# of trials * (n_samples, n_channels)]]
    """
    extraTimeLag = 200
    args = get_arg_parser()
    workspace = args.workspace
    example_data = torch.load(f"{workspace}/dynamic_trf_input_examples.pt")

    control_stims = example_data['control_stims']
    target_stims = example_data['target_stims']
    modulation_stims_full:NestedTensorDictList = example_data['modulation_stims']
    resps = example_data['resps']


    modulation_stims:NestedTensorDictList = [
        [
            StimDictTensor(
                   x = stims_full["x"][
                       np.array([
                           modulation_stim_names.index(name) 
                           for name in Config.modulation_stim_names
                    ])],
                   timeinfo = stims_full["timeinfo"]
               )
            for stims_full in stims_full_trials
        ] for stims_full_trials in modulation_stims_full
    ]
    

    default_configs = vars(args).copy()
    user_configs = {
        'contextModel': 'CausalConv',
        'fTRFMode': '+-a,b', #real value amplitude scaling (a) amd time shifit (b)
        'fs': 64,
        'workspace': workspace,
        'extraTimeLag': extraTimeLag,
        'device': 'cuda',
        'lr': (0.001, 0.01),
        'checkpoint': True,
        'folderName': f'dynamic_trf_2026_{mode.value}',
    }
    
    configs = default_configs.copy()
    configs.update(user_configs)

    configs = Configuration(**configs)

    assert configs.fs > 0

    if not Config.include_control:
        control_stims = []

    engine.run(control_stims, target_stims, modulation_stims, resps, configs)
