from typing import NamedTuple

import torch
import numpy as np

KFoldsReturn = NamedTuple(
    'KFoldsReturn',
    [
        ('idx_train', np.ndarray),
        ('idx_val', np.ndarray),
        ('idx_test', np.ndarray)
    ]
)


def k_folds(i_fold, n_trials, n_folds) -> KFoldsReturn:
    assert n_trials >= n_folds, (n_trials,n_folds)
    id_trials = np.arange(n_trials)
    splits = np.array_split(id_trials, n_folds)

    i_fold_val = (i_fold + 1) % n_folds
    idx_test = splits[i_fold]
    idx_val = splits[i_fold_val]
    idx_train = np.concatenate(
        [splits[i] for i in range(n_folds) if i not in [i_fold, i_fold_val]])

    return KFoldsReturn(
        idx_train = idx_train, 
        idx_val = idx_val, 
        idx_test = idx_test
    )

def count_parameters(model, ifName = False, oLog = None):
    if ifName:
        for name, param in model.named_parameters():
            if oLog is None:
                print(name, param.numel(), param)
            else:
                if name == '_model.oNonLinTRF.LinearKernels.nan.weight':
                    param = param.permute(2, 0, 1)
                elif name in ['_model.oNonLinTRF.oNonLinear.oEncoder.conv.weight', '_model.oNonLinTRF.oNonLinear.oEncoder.conv.bias','_model.oNonLinTRF.bias']:
                    torch.save(param.data, f'{name}.pth')
                

                oLog(name, param.numel(),param.shape, param)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         if oLog is None:
        #             print(p.numel())
        #         else:
        #             oLog(p.numel())

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parameters_will_optim(model, optimizer):
    # Extract optimizer parameters
    # Thanks to The DeepSeek R1 model give me the result below
    optimizer_params = []
    for group in optimizer.param_groups:
        optimizer_params.extend(group['params'])

    # Map model parameters to their names
    model_params = {param: name for name, param in model.named_parameters()}

    # Find tuned parameter names
    tuned_parameters = [model_params[param] for param in optimizer_params if param in model_params]

    print("Parameters being tuned:", tuned_parameters)
    return tuned_parameters