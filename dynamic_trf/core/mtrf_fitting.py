from typing import List, Any, Tuple

import torch
from mtrf.model import TRF
from mtrf.stats import neg_mse
from mtrf.stats import pearsonr as mtrf_pearsonr
from tour.dataclass.stim import to_impulses

from . import (
    Configuration, ScalarTensor
)
from ..utils.io import align_data

""" mTRFpy related starts """
def combine_control_target_stims(
    data:Tuple[List[Any], List[Any], List[Any]], fs:float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    control_stims, target_stims, resps = data
    target_stims = [to_impulses(
        x=tar['x'], timeinfo=tar['timeinfo'], f=fs
    ) for tar in target_stims]
    combined_stims = []
    if len(control_stims) > 0:
        for c_s, t_s in zip(control_stims, target_stims):
            combined_stims.append(
                torch.cat(align_data(c_s, t_s), dim = 0)
            )
    else:
        for t_s in target_stims:
            combined_stims.append(
                t_s
            )
    return combined_stims, resps
    
def trf_with_best_reg(
    train_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    val_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    configs: Configuration
)->Tuple[TRF, TRF, ScalarTensor, ScalarTensor, torch.Tensor]:
    # First discard the modulation stim
    train_data = tuple(train_data[i] for i in [0,1,3])
    val_data = tuple(val_data[i] for i in [0,1,3])

    ### combine the control stims and the target stims
    train_stim, train_resp = combine_control_target_stims(train_data, configs.fs)
    val_stim, val_resp = combine_control_target_stims(val_data, configs.fs)

    # here we need to transpose each array, because mtrf need sample dimension first array
    train_stim = [s.T for s in train_stim]
    train_resp = [r.T for r in train_resp]
    val_stim = [s.T for s in val_stim]
    val_resp = [r.T for r in val_resp]

    wds = 10.0 ** torch.arange(configs.lambda_range_power[0],configs.lambda_range_power[1]) #
    fs = configs.fs
    tmin,tmax = configs.timelag
    extraTimeLag = configs.extraTimeLag
    rs = torch.zeros((len(wds), train_resp[0].shape[-1]), dtype=train_resp[0].dtype)
    trfs = []
    
    for i, wd in enumerate(wds):
        trf = TRF(direction=1)#, metric=neg_mse)
        # train_stim = arrays_to_device(train_stim, 'cuda')
        # train_resp = arrays_to_device(train_resp, 'cuda')
        # print([(s.shape, r.shape) for s,r in zip(train_stim, train_resp)])
        trf.train(train_stim, train_resp, fs, tmin/1000, tmax/1000, regularization=wd)
        # train_stim = arrays_to_device(train_stim, 'cpu')
        # train_resp = arrays_to_device(train_resp, 'cpu')

        # val_stim = arrays_to_device(val_stim, 'cuda')
        # val_resp = arrays_to_device(val_resp, 'cuda')
        _,r = trf.predict(val_stim, val_resp, average=False)
        # val_stim = arrays_to_device(val_stim, 'cpu')
        # val_resp = arrays_to_device(val_resp, 'cpu')
        rs[i] = r
        trf.metric = mtrf_pearsonr
        trfs.append(trf)
        torch.cuda.empty_cache()

    rs_mean = rs.mean(1)
    sel_reg_idx = torch.argmax(rs_mean)
    bes_val_r_mean = torch.max(rs_mean)

    # fit the trf with the larget time lag
    trf_lrg = TRF(direction=1)
    trf_lrg.train(
        train_stim, train_resp, 
        fs, 
        (tmin - extraTimeLag)/1000, 
        (tmax + extraTimeLag)/1000, 
        wds[sel_reg_idx],
    )

    return trfs[sel_reg_idx], trf_lrg, wds[sel_reg_idx], bes_val_r_mean, rs[sel_reg_idx]

def test_mtrf_model(
    trf: TRF,
    test_data: Tuple[List[List[Any]], List[List[Any]], List[List[Any]], List[List[Any]]],
    configs: Configuration,
):
    # First discard the modulation stim
    test_data = list(test_data[i] for i in [0,1,3])
    if len(test_data[0]) == 0:
        test_data[0] = [[] for _ in test_data[1]]

    #based on the shape of the resps data
    # (n_subjs, n_trials_this_fold, n_resp_chans)
    rs = torch.zeros(
        (len(test_data[-1]), len(test_data[-1][0]), test_data[-1][0][0].shape[0]), 
        dtype=test_data[-1][0][0].dtype
    )
    for i_subj, t_test_data in enumerate(zip(*test_data)):
        test_stim, test_resp = combine_control_target_stims(t_test_data, configs.fs)
        test_stim = [s.T for s in test_stim]
        test_resp = [r.T for r in test_resp]
        # print(len(test_stim), test_stim[0].shape, len(test_resp), test_resp[0].shape)
        _,r = trf.predict(test_stim, test_resp, average=False)
        rs[i_subj] = r
    return rs

""" mTRFpy related ends """