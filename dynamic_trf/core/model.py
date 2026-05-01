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

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

""" start code written by GPT-5.2-Codex medium"""

@dataclass
class SampleResult:
	indices: np.ndarray
	points: np.ndarray
	labels: List[str]


def _validate_inputs(points: np.ndarray, labels: Sequence[str]) -> None:
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("points must be an array of shape (N, 2)")
	if len(labels) != points.shape[0]:
		raise ValueError("labels must have the same length as points")


def spatially_even_sample(
	points: np.ndarray,
	labels: Sequence[str],
	n_samples: int,
	seed: Optional[int] = None,
) -> SampleResult:
	"""
	Sample points in a spatially even way using a grid over the bounding box.

	The algorithm selects at most one point per grid cell, then fills or trims
	to reach n_samples.
	"""

	_validate_inputs(points, labels)

	if n_samples <= 0:
		raise ValueError("n_samples must be positive")
	if n_samples > points.shape[0]:
		raise ValueError("n_samples cannot exceed number of points")

	rng = np.random.default_rng(seed)
	grid_size = int(math.ceil(math.sqrt(n_samples)))

	x = points[:, 0]
	y = points[:, 1]
	x_min, x_max = np.min(x), np.max(x)
	y_min, y_max = np.min(y), np.max(y)

	# Expand bounds slightly to avoid edge effects on max values.
	eps = 1e-9
	x_max += eps
	y_max += eps

	x_bins = np.linspace(x_min, x_max, grid_size + 1)
	y_bins = np.linspace(y_min, y_max, grid_size + 1)

	x_idx = np.digitize(x, x_bins) - 1
	y_idx = np.digitize(y, y_bins) - 1

	cell_to_indices = {}
	for idx, (cx, cy) in enumerate(zip(x_idx, y_idx)):
		cell_to_indices.setdefault((cx, cy), []).append(idx)

	selected = []
	for cell_indices in cell_to_indices.values():
		selected.append(rng.choice(cell_indices))

	selected = np.array(selected, dtype=int)

	if selected.size > n_samples:
		selected = rng.choice(selected, size=n_samples, replace=False)
	elif selected.size < n_samples:
		remaining = np.setdiff1d(np.arange(points.shape[0]), selected)
		extra = rng.choice(remaining, size=n_samples - selected.size, replace=False)
		selected = np.concatenate([selected, extra])

	selected = np.sort(selected)
	sampled_points = points[selected]
	sampled_labels = [labels[i] for i in selected]

	return SampleResult(indices=selected, points=sampled_points, labels=sampled_labels)


def kmeans_sample(
	points: np.ndarray,
	labels: Sequence[str],
	n_samples: int,
	seed: Optional[int] = None,
	max_iters: int = 50,
) -> SampleResult:
	"""
	Sample points by k-means: choose nearest point to each centroid.
	"""

	_validate_inputs(points, labels)

	if n_samples <= 0:
		raise ValueError("n_samples must be positive")
	if n_samples > points.shape[0]:
		raise ValueError("n_samples cannot exceed number of points")

	rng = np.random.default_rng(seed)
	n_points = points.shape[0]

	init_indices = rng.choice(n_points, size=n_samples, replace=False)
	centroids = points[init_indices].copy()

	for _ in range(max_iters):
		# Assign points to nearest centroid.
		distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
		assignments = np.argmin(distances, axis=1)

		new_centroids = centroids.copy()
		for k in range(n_samples):
			cluster_points = points[assignments == k]
			if cluster_points.size == 0:
				new_centroids[k] = points[rng.integers(0, n_points)]
			else:
				new_centroids[k] = cluster_points.mean(axis=0)

		if np.allclose(new_centroids, centroids):
			centroids = new_centroids
			break
		centroids = new_centroids

	# Pick actual points nearest to centroids.
	distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
	nearest = np.argmin(distances, axis=0)
	selected = np.unique(nearest)

	if selected.size < n_samples:
		remaining = np.setdiff1d(np.arange(n_points), selected)
		extra = rng.choice(remaining, size=n_samples - selected.size, replace=False)
		selected = np.concatenate([selected, extra])
	elif selected.size > n_samples:
		selected = rng.choice(selected, size=n_samples, replace=False)

	selected = np.sort(selected)
	sampled_points = points[selected]
	sampled_labels = [labels[i] for i in selected]

	return SampleResult(indices=selected, points=sampled_points, labels=sampled_labels)

def plot_points_with_samples(
	points: np.ndarray,
	labels: Sequence[str],
	n_samples: int,
	seed: Optional[int] = None,
	show_labels: bool = True,
	method: str = "grid",
) -> Tuple[plt.Figure, SampleResult]:
	"""
	Plot all points, then highlight spatially even samples.
	"""

	if method == "grid":
		result = spatially_even_sample(points, labels, n_samples, seed=seed)
	elif method == "kmeans":
		result = kmeans_sample(points, labels, n_samples, seed=seed)
	else:
		raise ValueError("method must be 'grid' or 'kmeans'")

	fig, ax = plt.subplots(figsize=(12, 8))
	ax.scatter(points[:, 0], points[:, 1], s=18, alpha=0.35, label="All points")
	ax.scatter(
		result.points[:, 0],
		result.points[:, 1],
		s=60,
		c="#e45756",
		edgecolor="white",
		linewidth=0.7,
		label="Sampled",
	)

	if show_labels:
		for (x, y), text in zip(result.points, result.labels):
			ax.annotate(text, (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")

	ax.set_xlabel("amplitude scaling")
	ax.set_ylabel("time-shifting")
	ax.legend(frameon=False)
	ax.set_title("distribution of transformation parameters")
	fig.tight_layout()

	return fig, result

def _example_usage() -> None:
	rng = np.random.default_rng(7)
	points = rng.normal(size=(200, 2))
	labels = [f"p{i}" for i in range(points.shape[0])]

	plot_points_with_samples(
		points,
		labels,
		n_samples=20,
		seed=42,
		show_labels=True,
		method="kmeans",
	)
	plt.show()


""" end code written by GPT-5.2-Codex medium"""

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
        if len(model.trfs) > 1:
            astrf:ASTRF = model.trfs[1]
            feats_key = model.feats_keys[1]
        else:
            astrf:ASTRF = model.trfs[0]
            feats_key = model.feats_keys[0]
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
            tag_0 = feats[0]['tag']
            xs = []
            for feat in feats:
                xs.append(feat['x'])
                torch.equal(timeinfo_0, feat['timeinfo'])
            xs = torch.cat(xs, dim = -2)
            feats = {
                'x':xs,
                'timeinfo':timeinfo_0,
                'tag':tag_0
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
            plt.title(f'transfomrer weights {feats_key[0]}-{iIn}')
            figures.append(fig)
        
        aSeq, bSeq, cSeq = astrf.trfsGen.getTransformParams(feats['x'], None)
        aSeq = aSeq.detach().squeeze().cpu().numpy()
        bSeq = bSeq.detach().squeeze().cpu().numpy()
        # fig = plt.figure()
        # plt.scatter(aSeq, bSeq)
        tags = []
        for tag1, tag2 in zip([''] + feats['tag'][:-1], feats['tag']):
              tags.append(f"{tag1},{tag2}")
            
        fig, result = plot_points_with_samples(
            points = np.stack([aSeq, bSeq], axis = 1),
            labels = tags,
            n_samples=30,
            seed=42,
            show_labels=True,
            method="kmeans",
        )
        figures.append(fig)

        x = feats['x'].detach().cpu().numpy()
        # print(x.shape)
        x_prev = x[0, 0, result.indices-1]
        x_current = x[0, 0, result.indices]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x_prev, x_current, s=18, alpha=0.35, label="All points")
        ax.scatter(
            x_prev,
            x_current,
            s=60,
            c="#e45756",
            edgecolor="white",
            linewidth=0.7,
            label="Sampled",
        )
        for x_, y_, text in zip(x_prev, x_current, result.labels):
            ax.annotate(text, (x_, y_), fontsize=9, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("prior modulation parameter")
        ax.set_ylabel("current modulation parameter")
        ax.legend(frameon=False)
        ax.set_title("distribution of modulation parameter")
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
            if len(model.trfs) > 1:
                cnntrf:CNNTRF = model.trfs[0]
                astrf:ASTRF = model.trfs[1]
            else:
                cnntrf = None
                astrf:ASTRF = model.trfs[0]
            
            # plot dynamic TRFs
            curFigs1 = self.plot_trfs(model)

            # plot linear kernel of ASTRF
            curFigs = self.plot_ltitrf(astrf) 
            figures.extend(curFigs1)
            figures.extend(curFigs)
            
            if cnntrf is not None:
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
