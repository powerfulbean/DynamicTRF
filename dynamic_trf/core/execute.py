from typing import List, Any, Tuple

import torch
import numpy as np

from tqdm import tqdm
from mtrf.model import TRF
from matplotlib import pyplot as plt


from tour.dataclass.stim import to_impulses
from tour.torch_trainer import Context, StimRespDataset, SaveBest, BatchAccumulator, get_logger
from ..utils import count_parameters, k_folds
from ..utils.io import pickle_save, align_data, arrays_to_device
from . import (
    torchdata, NestedTensorList, NestedTensorDictList, Configuration,
    flatten_nested_list, ScalarTensor
)

from tray.stats import plot_biosemi128

from dynamic_trf.core.model import (
    TwoMixedTRF, 
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
    logger = get_logger(configs.tarDir)
    logger.info('start train step of dynamic trf')
    n_folds = Configuration.nFolds
    for i_fold in tqdm(range(n_folds), desc='cross validation', leave=False):
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
                None, configs.tarDir, units = 'a.u.', tmin = t_trf.times[0].cpu().numpy())
            
            plot_biosemi128(
                t_trf_lrg.weights[i].T, f'larger lag fold{i_fold} weights {i}', 
                None, configs.tarDir, units = 'a.u.', tmin = t_trf_lrg.times[0].cpu().numpy())
        
        plot_biosemi128(t_r, f'fold{i_fold} r', None, configs.tarDir, units = 'r')

        logger.info(f"selected lambda is: {t_wd}, the best validation prediction r is: {t_mean_r}")

        
        mtrf_test_rs = test_mtrf_model(t_trf, test_data, configs)
        print(mtrf_test_rs.shape)

        if not configs.mtrf_only:
            train_step(t_trf, t_trf_lrg, train_data, val_data, configs, configs.randomSeed)
        
def train_step(
    trf:TRF,
    trf_lrg:TRF,
    train_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    val_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
    configs: Configuration,
    seed = 42, 
):
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
    logger = get_logger(configs.tarDir)
    logger.info('train step of dynamic trf start')

    trainerDir = configs.tarDir
    srate = configs.fs
    device = configs.device
    batchSize = configs.batchSize
    optimStr = configs.optimizer
    minLr,maxLr = configs.lr
    wd1 = configs.wd

    train_torch_ds = torchdata.TorchDataset(*train_data, device = device)
    val_torch_ds = torchdata.TorchDataset(*val_data, device=device)

    stim_dict_tensor_old, resp = torchdata.TorchDataset(*train_data, device = device)[0]
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
    
    train_torch_dl = torch.utils.data.DataLoader(train_torch_ds)
    test_torch_dl = torch.utils.data.DataLoader(val_torch_ds)

    linW = trf.weights
    linB = trf.bias
    linW_lrgrLag = trf_lrg.weights
    
    dim_info = dict(
        linInDim = linInDim,
        nonlinInDim = nonlinInDim,
        auxInDim = auxInDim,
        outDim = outDim,
    )

    mixed_rf:TwoMixedTRF = build_mixed_model(**dim_info, configs=configs)
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
    dldr = torch.utils.data.DataLoader(torchdata.TorchDataset(*train_data,device = device),batch_size = 1)
    nnTRFInput = next(iter(dldr))
    # print(len(nnTRFInput), len(nnTRFInput[0]), len(nnTRFInput[1]))
    # print(mTRFpyInput.shape)
    predTRFpy = trf.predict(mTRFpyInput)[0].cpu().numpy()
    # print(oMixedRF.parseBatch(nnTRFInput)[0].shape)
    real_feats_keys = mixed_rf.feats_keys
    mixed_rf.feats_keys = [[torchdata.CONTROL_STIM_TAG], [torchdata.TARGET_STIM_TAG]]

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
    
    stop
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
    cycleIter = (len(datasets['train']) // batchSize) * 2
    if lrScheduler == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,minLr,maxLr,cycleIter,mode = 'triangular2',cycle_momentum=False)
    elif lrScheduler is None:
        lr_scheduler = None
    elif lrScheduler == 'reduce':
        assert minLr == maxLr
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 4)
    else:
        raise NotImplementedError()

    oTrainer = CTrainer(epoch, device, criterion, optimizer,lr_scheduler)
    oTrainer.setDir(oLog,trainerDir)
    oTrainer.setDataLoader(dataloaders['train'],dataloaders['dev'])
    fPlot = PlotInterm(srate,sample_batch)
    oTrainer.addPlotFunc(fPlot)
    
    metricPerson = CMPearsonr(output_transform=fPickPredTrueFromOutputT,avgOutput = False)
    oTrainer.addMetrics('corr', metricPerson)
    bestEpoch,bestDevMetrics= oTrainer.train(oMixedRF,'corr',
                                        trainingStep=CTrainForwardFunc,
                                        evaluationStep=CEvalForwardFunc,
                                        patience = 10)
    pickle_save(model_config,oTrainer.tarFolder + '/configs.bin')
    pickle_save(bestDevMetrics,oTrainer.tarFolder + '/devMetrics.bin')
    bestModel:TwoMixedTRF = from_pretrainedMixedRF(model_config, oTrainer.tarFolder + '/savedModel_feedForward_best.pt')        
    bestModel.trfs[1].if_enable_trfsGen = True
    bestModel.trfs[1].stop_update_linear()
    bestModel.eval()

    #assert the linear part is not changed
    newWB = getLinModelWB(bestModel)
    assert all([np.array_equal(cachedWB[i], newWB[i]) for i in range(len(cachedWB))])

    logger.info(f"train step of dynamic trf complete")
    oTrainer.trainer = None
    oTrainer.evaluator = None
    oTrainer.model = None
    oTrainer.optimizer = None
    oTrainer.lrScheduler = None
    oTrainer.oLog = None
    oTrainer.dtldTrain = None
    oTrainer.dtldDev = None
    oTrainer.fPlotsFunc = None
    # stop
    return None,model_config,oTrainer,bestEpoch,bestDevMetrics,trainerDir
