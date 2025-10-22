# from StimRespFlow.DataProcessing.DeepLearning.Trainer import CTrainerFunc
from typing import Tuple, Dict

import torch
import numpy as np

from torch.nn.functional import pad
from matplotlib import pyplot as plt

from nntrf import models as nntrf_models
from nntrf.models import CNNTRF, ASTRF, FuncTRFsGen, MixedTRF, LTITRFGen, msec2Idxs, Idxs2msec, TRFAligner

from . import Configuration
from .data import (
    CONTROL_STIM_TAG,
    MODULATION_STIM_TAG,
)


class PlotInterm:
    
    def __init__(self,srate, sample_batch):
        self.srate = srate
        self.sample_batch = sample_batch
        self.cnter = 0
    
    def plot_cnntrf(self,cnntrf:CNNTRF):
        times = cnntrf.lagTimes
        figures = []
        for i in range(cnntrf.weights.shape[1]):
            fig2 = plt.figure()
            plt.plot(times,cnntrf.weights[:,i,:].T)
            figures.append(fig2)
        return figures

    def plot_trfs(self,model:MixedTRF):
        figures = []
        fig = plt.figure()
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        astrf:ASTRF = model.trfs[1]
        feats_key = model.feats_keys[1]
        feats = []
        feat_dict,_ = self.sample_batch
        for feat_key in feats_key:
            # print(feat_dict.keys())
            feat = feat_dict[feat_key]
            assert isinstance(feat, dict)
            feats.append(feat)
            # concatente
        if len(feats) == 1:
            feats = feats[0]
        else:
            # raise NotImplementedError
            timeinfo_0 = feats[0]['timeinfo']
            xs = []
            for feat in feats:
                xs.append(feat['x'])
                torch.equal(timeinfo_0, feat['timeinfo'])
            xs = torch.cat(xs, dim = -2)
            feats = {
                'x':xs,
                'timeinfo':timeinfo_0
            }

        # (nBatch, outDim, nWin, nSeq)
        trfs = astrf.get_trfs(feats['x'])
        assert trfs.shape[0] == 1
        trfs = trfs[0].permute(2,1,0).cpu().numpy()

        for idx,TRF in enumerate(trfs):
            if TRF.shape[1] == 128:
                tarTRF = TRF[:,18]
            else:
                tarTRF = TRF[:,0:1]
            plt.plot(astrf.lagTimes,tarTRF,color = cycle[idx % len(cycle)])
            # break
        figures.append(fig)


        ws = astrf.trfsGen.transformer.conv.weight.cpu()
        for iIn in range(ws.shape[1]):
            fig = plt.figure()
            plt.plot(ws[:,iIn,:].numpy().T, marker='o')
            plt.title(f'transfomrer weights {feats_key[iIn]}')
            figures.append(fig)
        return figures


    def plot_ltitrf(self,astrf:ASTRF):
        times = astrf.lagTimes
        figures = []
        weight = astrf.ltiTRFsGen.weight.cpu().detach().numpy()
        inDim = weight.shape[1]
        for i in range(inDim):
            fig = plt.figure()
            weight = weight[:,i,:].T
            plt.plot(times,weight) #
            figures.append(fig) 
        return figures

    def __call__(self,model:MixedTRF):
        self.cnter += 1
        figures = []
        with torch.no_grad():
            model.eval()
            cnntrf:CNNTRF = model.trfs[0]
            astrf:ASTRF = model.trfs[1]
            
            astrf.lagTimes
            
            # plot dynamic TRFs
            curFigs1 = self.plot_trfs(model)

            # plot linear kernel of ASTRF
            curFigs = self.plot_ltitrf(astrf) 
            figures.extend(curFigs1)
            figures.extend(curFigs)
            
            figs = self.plot_cnntrf(cnntrf)
            figures.extend(figs)
    
        return figures

def func_forward(model:torch.nn.Module, batch:Tuple[Dict[str, torch.Tensor], torch.Tensor]):
    pred,y = model(*batch)
    return pred, y

def seqLast_pad_zero(seq):
    maxLen = max([i.shape[-1] for i in seq])
    output = []
    for i in seq:
        output.append(pad(i,(0,maxLen - i.shape[-1])))
    return torch.stack(output,0)

def collate_fn_CMixedTRF(samples):
    stimKeys = samples[0][0].keys()
    stims = {k:[] for k in stimKeys}
    resps = []
    infos = {k:[] for k in samples[0][2].keys()}
    for smpl in samples:
        s,r,info = smpl
        for k in stimKeys:
            stims[k].append(s[k])
        resps.append(r)
        for k in infos:
            infos[k].append(info[k])
    resps = seqLast_pad_zero(resps)
    for k,v in stims.items():
        if isinstance(v[0], torch.Tensor) and k != 'tIntvl':
            stims[k] = seqLast_pad_zero(v)
    #pad1 for stim key except 'vector' and 'tIntvl'
    #pad2 for 'vector' only,check if len(transforms) == len(tIntvl) in oneofbatch
    return stims,resps,infos


def build_mixed_model(
    linInDim,
    nonlinInDim,
    auxInDim,
    outDim,
    configs: Configuration
) -> MixedTRF: 

    tmin_ms, tmax_ms = configs.timelag
    fs = configs.fs
    device = configs.device
    contextModel = configs.contextModel
    mode = configs.fTRFMode
    nNonLinWin = configs.nContextWin
    nBasis = configs.nBasis
    limitOfShift_idx = configs.limitOfShift_idx

    nTransParams = len(FuncTRFsGen.parse_trans_params(mode))
    control_stim_tag = CONTROL_STIM_TAG
    modulation_stim_tag = MODULATION_STIM_TAG

    trfs_list = []
    feattags_list = []
    if linInDim > 0:
        trf1 = CNNTRF(
            linInDim,
            outDim,
            tmin_ms,
            tmax_ms,
            fs
        )
        trfs_list.append(trf1)
        feattags_list.append([control_stim_tag])

    #module that estimates transformation parameter
    if isinstance(contextModel, str):
        context_model = getattr(nntrf_models, contextModel)(nonlinInDim + auxInDim, nTransParams, nNonLinWin)
    elif isinstance(contextModel, torch.nn.Module):
        context_model = contextModel
    else:
        raise ValueError('context_model should be a module or str')

    trfsGen = FuncTRFsGen(
        nonlinInDim, 
        outDim, 
        tmin_ms, 
        tmax_ms, 
        fs, 
        basisTRFName='gauss', 
        limitOfShift_idx=limitOfShift_idx, 
        nBasis = nBasis,
        mode = mode,
        transformer = context_model,
        device = device
    )

    trf2 = ASTRF(
        nonlinInDim, 
        outDim, 
        tmin_ms, 
        tmax_ms, 
        fs, 
        trfsGen = trfsGen,
        device = device,
        verbose = False
    )
    
    trfs_list.append(trf2)
    feattags_list.append([modulation_stim_tag])

    mixedRF = MixedTRF(
        device,
        trfs_list,
        feattags_list
    ).to(device).to(torch.get_default_dtype())
    return mixedRF

def from_pretrainedMixedRF(configs, state_dict, cpu = False):
    oMixedRF = build_mixed_model(**configs)
    if isinstance(state_dict,str):
        if cpu:
            oMixedRF.load_state_dict(
                torch.load(state_dict,map_location=torch.device('cpu'))['model_state_dict'])
        else:
            oMixedRF.load_state_dict(torch.load(state_dict)['model_state_dict'])
    else:
        oMixedRF.load_state_dict(state_dict)
    return oMixedRF
