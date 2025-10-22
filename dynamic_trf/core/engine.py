from typing import List, Any, Tuple

import torch
import numpy as np

from tqdm import tqdm
from mtrf.model import TRF
from matplotlib import pyplot as plt


from tour.dataclass.stim import to_impulses
from tour.torch_trainer import Context, SaveBest, BatchAccumulator, get_logger, pearsonr

from .model import func_forward, from_pretrainedMixedRF
from ..utils import count_parameters, k_folds
from ..utils.io import align_data, checkFolder
from . import (
    data, NestedTensorList, NestedTensorDictList, Configuration,
    flatten_nested_list, ScalarTensor
)

from tray.stats import plot_biosemi128

from dynamic_trf.core.model import (
    MixedTRF, 
    ASTRF, 
    CNNTRF, 
    build_mixed_model, 
    from_pretrainedMixedRF,
    PlotInterm,
)

""" mTRFpy related starts """
def combine_control_target_stims(
    data:Tuple[List[Any], List[Any], List[Any]], fs:float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    control_stims, target_stims, resps = data
    target_stims = [to_impulses(
        x=tar['x'], timeinfo=tar['timeinfo'], f=fs
    ) for tar in target_stims]
    combined_stims = []
    for c_s, t_s in zip(control_stims, target_stims):
        combined_stims.append(
            torch.cat(align_data(c_s, t_s), dim = 0)
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

    wds = 10.0 ** torch.arange(-3, -2) #-5,6
    fs = configs.fs
    tmin,tmax = configs.timelag
    extraTimeLag = configs.extraTimeLag
    rs = torch.zeros((len(wds), train_resp[0].shape[-1]), dtype=train_resp[0].dtype)
    trfs = []
    
    for i, wd in enumerate(wds):
        trf = TRF(direction=1)
        # train_stim = arrays_to_device(train_stim, 'cuda')
        # train_resp = arrays_to_device(train_resp, 'cuda')
        trf.train(train_stim, train_resp, fs, tmin/1000, tmax/1000, regularization=wd)
        # train_stim = arrays_to_device(train_stim, 'cpu')
        # train_resp = arrays_to_device(train_resp, 'cpu')

        # val_stim = arrays_to_device(val_stim, 'cuda')
        # val_resp = arrays_to_device(val_resp, 'cuda')
        _,r = trf.predict(val_stim, val_resp, average=False)
        # val_stim = arrays_to_device(val_stim, 'cpu')
        # val_resp = arrays_to_device(val_resp, 'cpu')
        rs[i] = r
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
    test_data = tuple(test_data[i] for i in [0,1,3])

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
        _,r = trf.predict(test_stim, test_resp, average=False)
        rs[i_subj] = r
    return rs

""" mTRFpy related ends """

def split_data_for_subject(
    data:List[List[Any]], i_fold:int, n_folds:int) -> Tuple[List[List[Any]],List[List[Any]],List[List[Any]]]:
    train_data = []
    val_data = []
    test_data = []

    f_select_data = lambda x, idxs: [x[idx] for idx in idxs]
    for i_subj_data in data:
        n_trials = len(i_subj_data)
        k_folds_return = k_folds(i_fold, n_trials, n_folds)
        idx_train, idx_val, idx_test = k_folds_return
        train_data.append(
            f_select_data(i_subj_data, idx_train)
        )
        val_data.append(
            f_select_data(i_subj_data, idx_val)
        )
        test_data.append(
            f_select_data(i_subj_data, idx_test)
        )
    return train_data, val_data, test_data


def run(
    control_stims:NestedTensorList, 
    target_stims:NestedTensorDictList, 
    modulation_stims:NestedTensorDictList,
    resps:NestedTensorList, 
    configs:Configuration
):
    """
    Parameters:
    ------------
    control_stims: NestedTensorList
        the timeseries of control stimuli,
        nested List of tensor each item of the outer list corresponding to one subject, each item of the inner list corresponding to one trial
        the size of it is [# of subject * [# of trials * (n_samples, n_channels)]]
    target_stims: NestedTensorDictList
        the dict of target stimuli
        nested List of tensor each item of the outer list corresponding to one subject, each item of the inner list corresponding to one trial
        the size of it is [# of subject * [# of trials * StimDict]]
    resps: NestedTensorList
        the brain responses
    configs: dict
        configuration parameters of dynamic trf
    """
    logger = get_logger(configs.tarDir, if_print=True)
    logger.info('dynamic trf analysis started')
    logger = get_logger(configs.tarDir, if_print=False)
    n_folds = Configuration.nFolds
    for i_fold in tqdm(range(n_folds), desc='cross validation', leave=False):
        t_tar_dir = f'{configs.tarDir}/{i_fold}'
        checkFolder(t_tar_dir)
        logger.info(f"start the {i_fold}th fold")
        control_stims_splited= split_data_for_subject(
            control_stims,
            i_fold,
            n_folds
        )
        target_stims_splited= split_data_for_subject(
            target_stims,
            i_fold,
            n_folds
        )
        modulation_stims_splited = split_data_for_subject(
            modulation_stims,
            i_fold,
            n_folds
        )
        resps_splited= split_data_for_subject(
            resps,
            i_fold,
            n_folds
        )
        
        train_data, val_data, test_data = list(zip(
            control_stims_splited,
            target_stims_splited,
            modulation_stims_splited,
            resps_splited,
        ))

        train_data = [flatten_nested_list(d) for d in train_data]
        val_data = [flatten_nested_list(d) for d in val_data]

        t_trf, t_trf_lrg, t_wd, t_mean_r, t_r = trf_with_best_reg(
            train_data=train_data, val_data=val_data, configs=configs)
    
        for i in range(t_trf.weights.shape[0]):
            plot_biosemi128(
                t_trf.weights[i].T, f'fold{i_fold} weights {i}', 
                None, t_tar_dir, units = 'a.u.', tmin = t_trf.times[0].cpu().numpy())
            
            plot_biosemi128(
                t_trf_lrg.weights[i].T, f'larger lag fold{i_fold} weights {i}', 
                None, t_tar_dir, units = 'a.u.', tmin = t_trf_lrg.times[0].cpu().numpy())
        
        plot_biosemi128(t_r, f'fold{i_fold} r', None, t_tar_dir, units = 'r')

        logger.info(f"selected lambda is: {t_wd}, the best validation prediction r is: {t_mean_r}")

        
        mtrf_test_rs = test_mtrf_model(t_trf, test_data, configs)
        # print(mtrf_test_rs.shape)

        if not configs.mtrf_only:
            mixed_rf = train_step(t_trf, t_trf_lrg, train_data, val_data, configs, t_tar_dir, configs.randomSeed)

            dytrf_test_rs = test_model(mixed_rf, test_data, configs, t_tar_dir)

        logger = get_logger(configs.tarDir, if_print=True)
        logger.info('dynamic trf analysis completed')

        print(mtrf_test_rs.shape, dytrf_test_rs.shape, mtrf_test_rs.mean(), dytrf_test_rs.mean())

def test_model(
    mixed_rf: MixedTRF,
    test_data: Tuple[List[List[Any]], List[List[Any]], List[List[Any]], List[List[Any]]],
    configs: Configuration,
    folder:str,
):
    
    trainerDir = folder
    srate = configs.fs
    device = configs.device
    batchSize = configs.batchSize
    optimStr = configs.optimizer
    minLr,maxLr = configs.lr
    wd1 = configs.wd
    lrScheduler = configs.lrScheduler

    f_mse = torch.nn.MSELoss()
    def func_metrics(batch, output:Tuple[torch.Tensor, torch.Tensor]):
        pred, y = output

        r = pearsonr(
            y.transpose(-1, -2),
            pred.transpose(-1, -2)
        )
        return {
            'loss': f_mse(y, pred),
            'r_avg': r.mean(),
            'r': r 
        }

    trainer_ctx = Context(
        mixed_rf,
        None,
        func_metrics,
        trainerDir,
        if_print_metric=False
    )

    # (n_subjs, n_trials_this_fold, n_resp_chans)
    rs = torch.zeros(
        (len(test_data[-1]), len(test_data[-1][0]), test_data[-1][0][0].shape[0]), 
        dtype=test_data[-1][0][0].dtype
    )

    def func_cat(values):
        # print(torch.cat(values).shape)
        if values[0].ndim == 0:
            return torch.stack(values)
        else:
            return torch.cat(values)

    for i_subj, t_test_data in enumerate(zip(*test_data)):
        test_torch_ds = data.TorchDataset(*t_test_data, device=device)
        test_torch_dl = torch.utils.data.DataLoader(test_torch_ds, batch_size=1)
        metrics, _ = trainer_ctx.evaluate_dataloader(
            'test', test_torch_dl, func_forward, f_reduce_metrics_records=func_cat
        )
        metrics = metrics['test/r']
        # print(metrics.shape)
        rs[i_subj] = metrics
    return rs

def train_step(
    trf:TRF,
    trf_lrg:TRF,
    train_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    val_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    configs: Configuration,
    folder:str,
    seed = 42, 
) -> MixedTRF:
    checkFolder(folder)
    #set torch random_seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    linInDim = train_data[0][0].shape[0]
    nonlinInDim = train_data[1][0]['x'].shape[0]
    auxInDim = train_data[2][0]['x'].shape[0] - nonlinInDim
    outDim = train_data[3][0].shape[0]

    #get a sample test input for validating consistency between trf and nntrf
    mTRFpyInput = combine_control_target_stims(
        [train_data[i] for i in [0,1,3]], 
        configs.fs,
    )[0][0].T

    # oLog = None
    logger = get_logger(configs.tarDir, if_print=False)
    logger.info('train step of dynamic trf start')

    trainerDir = folder
    srate = configs.fs
    device = configs.device
    batchSize = configs.batchSize
    optimStr = configs.optimizer
    minLr,maxLr = configs.lr
    wd1 = configs.wd
    lrScheduler = configs.lrScheduler

    train_torch_ds = data.TorchDataset(*train_data, device = device)
    val_torch_ds = data.TorchDataset(*val_data, device=device)

    stim_dict_tensor_old, resp = data.TorchDataset(*train_data, device = device)[0]
    resp = resp.clone()[None,...]
    stim_dict_tensor = {}
    for k in stim_dict_tensor_old:
        feat = stim_dict_tensor_old[k]
        if isinstance(feat, dict):
            stim_dict_tensor[k] = {k2:feat[k2].clone()[None, ...] for k2 in feat}
        else:
            stim_dict_tensor[k] = feat.clone()[None, ...]

    
    sample_batch = (stim_dict_tensor, resp)
    assert batchSize == 1
    

    linW = trf.weights
    linB = trf.bias
    linW_lrgrLag = trf_lrg.weights
    
    dim_info = dict(
        linInDim = linInDim,
        nonlinInDim = nonlinInDim,
        auxInDim = auxInDim,
        outDim = outDim,
    )

    mixed_rf:MixedTRF = build_mixed_model(**dim_info, configs=configs)
    logger.info(f"{mixed_rf}")
    logger.info(f'number of trainable parameters: {count_parameters(mixed_rf)}')#,True, oLog))
    #additionaly we also fit a linear TRF with larger time lag for non-linear shifting of TRF
    
    cnntrf:CNNTRF = mixed_rf.trfs[0]
    cnntrf.loadFromMTRFpy(linW[0:linInDim], linB/2,device)

    astrf:ASTRF = mixed_rf.trfs[1]
    astrf.trfsGen.fitFuncTRF(linW_lrgrLag[-nonlinInDim:])
    astrf.set_linear_weights(linW[-nonlinInDim:], linB/2)
    astrf.if_enable_trfsGen = False
    astrf.stop_update_linear()
    fig = astrf.trfsGen.basisTRF.vis()
    fig.savefig(f'{trainerDir}/visFTRF.png')
    plt.close(fig)
    
    # print(linW.shape)

    def getLinModelWB(oModel):
        cachedW1 = oModel.trfs[0].oCNN.weight.detach().cpu().numpy()
        cachedB1 = oModel.trfs[0].oCNN.bias.detach().cpu().numpy()
        cachedW2 = oModel.trfs[1].ltiTRFsGen.weight.detach().cpu().numpy()
        return cachedW1,cachedB1, cachedW2

    cachedWB = getLinModelWB(mixed_rf)

    #validate the results are almost the same
    dldr = torch.utils.data.DataLoader(data.TorchDataset(*train_data,device = device),batch_size = 1)
    nnTRFInput = next(iter(dldr))
    # print(len(nnTRFInput), len(nnTRFInput[0]), len(nnTRFInput[1]))
    # print(mTRFpyInput.shape)
    predTRFpy = trf.predict(mTRFpyInput)[0].cpu().numpy()
    # print(oMixedRF.parseBatch(nnTRFInput)[0].shape)
    real_feats_keys = mixed_rf.feats_keys
    mixed_rf.feats_keys = [[data.CONTROL_STIM_TAG], [data.TARGET_STIM_TAG]]

    # control_stim_1 = mTRFpyInput[:,:2]
    # control_stim_2 = nnTRFInput[0][torchdata.CONTROL_STIM_TAG]
    # mtrf_sub = trf.copy()
    # mtrf_sub.weights = trf.weights[:2].copy()
    # mtrf_sub.bias = trf.bias.copy()/2

    # with torch.no_grad():
    #     control_stim_1_pred = mtrf_sub.predict(control_stim_1)[0]
    #     control_stim_2_pred = cnntrf(control_stim_2)[0].cpu().numpy()

    # print(control_stim_1_pred.shape, control_stim_2_pred.shape)
    # assert np.allclose(control_stim_1.T, control_stim_2.cpu().numpy())
    # assert np.allclose(control_stim_1_pred.T, control_stim_2_pred),\
    #       np.abs(control_stim_1_pred.T - control_stim_2_pred).max()

    predNNTRFOutput = mixed_rf(*nnTRFInput)
    predNNTRF = predNNTRFOutput[0].detach().cpu().numpy()[0].T
    # predNNTRF = predNNTRF[...,:1000,:]
    # predTRFpy = predTRFpy[...,:1000,:]
    # print(predTRFpy.shape, predNNTRF.shape)
    # print(predTRFpy, predNNTRF)
    assert np.allclose(
        predNNTRF,predTRFpy),\
        (   
            np.abs(predNNTRF - predTRFpy).max(),
            np.abs(predNNTRF - predTRFpy).argmax(),
        )
    #enable non-linear
    mixed_rf.feats_keys = real_feats_keys
    astrf.if_enable_trfsGen = True
    

    criterion = torch.nn.MSELoss()
    params_for_train = astrf.get_params_for_train()
    if optimStr == 'AdamW':
        optimizer = torch.optim.AdamW(
            params = params_for_train,
            lr = minLr,
            weight_decay = wd1
        )
    elif optimStr == 'AdamW-amsgrad':
        optimizer = torch.optim.AdamW(
            params = params_for_train,
            lr = minLr,
            weight_decay = wd1, amsgrad = True
        )
    elif optimStr == 'Adam':
        optimizer = torch.optim.Adam(
            params = params_for_train,
            lr = minLr,
            weight_decay = wd1, amsgrad = False
        )
    else:
        raise NotImplementedError()

    # oMixedRF.oNonLinTRF.stopUpdateLinear()
    astrf.stop_update_linear()
    cycleIter = (len(train_torch_ds) // batchSize) * 2
    if lrScheduler == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,minLr,maxLr,cycleIter,mode = 'triangular2',cycle_momentum=False)
    elif lrScheduler is None:
        lr_scheduler = None
    elif lrScheduler == 'reduce':
        assert minLr == maxLr
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 4)
    else:
        raise NotImplementedError()
    
    f_mse = torch.nn.MSELoss()

    def func_metrics(batch, output:Tuple[torch.Tensor, torch.Tensor]):
        pred, y = output

        r = pearsonr(
            y.transpose(-1, -2),
            pred.transpose(-1, -2)
        )
        return {
            'loss': f_mse(y, pred),
            'r_avg': r.mean(),
            'r': r 
        }
    
    trainer_ctx = Context(
        mixed_rf,
        optimizer,
        func_metrics,
        trainerDir,
        if_print_metric=False,
    )
    save_best = SaveBest(trainer_ctx,'val/r_avg', lambda old, new: old < new, tol = 20)
    func_plot = PlotInterm(srate,sample_batch)
    train_torch_dl = torch.utils.data.DataLoader(train_torch_ds, batch_size=1)
    val_torch_dl = torch.utils.data.DataLoader(val_torch_ds, batch_size=1)

    progress_bar = tqdm(range(configs.epoch), desc = "training", leave = False, dynamic_ncols=True)
    for i_epoch in progress_bar:
        trainer_ctx.new_epochs()
        for batch in tqdm(train_torch_dl, desc = "batch", leave=False):
            optimizer.zero_grad()
            mixed_rf.train()
            output = func_forward(mixed_rf, batch)
            loss = f_mse(*output)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        collab_metrics = {}

        _, train_metrics = trainer_ctx.evaluate_dataloader(
            'train', 
            train_torch_dl,
            func_forward,
            save_in_context=True
        )

        _, val_metrics = trainer_ctx.evaluate_dataloader(
            'val', 
            val_torch_dl,
            func_forward,
            save_in_context=True
        )

        collab_metrics.update(train_metrics)
        collab_metrics.update(val_metrics)
        progress_bar.set_postfix(collab_metrics)

        metrics_to_log = {k:trainer_ctx.metrics_log[k][-1] for k in trainer_ctx.metrics_log}
        # trainer_ctx.logger.info(f"epoch-{i_epoch}-{metrics_to_log}")

        ifUpdate, ifStop = save_best.step()
        if ifUpdate:
            fPlot:List[plt.Figure] = func_plot(mixed_rf)
            for i_fig, fig in enumerate(fPlot):
                fig.savefig(f"{trainerDir}/{i_fig}.png")
                plt.close(fig)
        
        if ifStop:
            break

    logger.info(f"train step of dynamic trf complete")

    model_configs = dict(
        configs = configs
    )

    model_configs.update(dim_info)

    saved_mixed_rf = from_pretrainedMixedRF(
        model_configs,
        save_best.target_path,
    )

    return saved_mixed_rf
