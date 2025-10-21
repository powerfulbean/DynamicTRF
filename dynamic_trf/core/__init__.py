
from typing import List, Dict, TypeVar, Generic, Any, Annotated
from dataclasses import dataclass, asdict
from itertools import chain

import torch
import numpy as np

from ..utils.io import checkFolder

Array = TypeVar('Array')
@dataclass
class StimDict(Generic[Array]):
    x: Array
    timeinfo: Array

@dataclass
class Configuration:
    folderName: str = 'dynamic_trf'
    tarDirRoot: str = './'
    mtrf_only: bool = False
    contextModel: str = 'CausalConv'
    nContextWin: int = 2
    fTRFMode: str = '+-a,b'
    nBasis: int = 21
    timelag: List[int] = (0, 700)
    nFolds: int = 10
    epoch: int = 100
    batchSize: int = 1
    wd: float = 0.01
    lr: List[float]= (0.001,0.001)
    optimizer: str = 'AdamW'
    lrScheduler: str = 'cycle'
    randomSeed: int = 42
    fs: int = -1
    extraTimeLag: int = 200
    device:str = 'cpu'

    @property
    def tarDir(self):
        out = self.tarDirRoot + '/' + self.folderName
        checkFolder(out)
        return out
    
    @property
    def limitOfShift_idx(self):
        return int(np.ceil(self.fs * self.extraTimeLag/1000))

ScalarTensor = Annotated[torch.Tensor, "scalar (shape=())"]

StimDictArray = StimDict[np.ndarray]
StimDictTensor = StimDict[torch.Tensor]


NestedArrayList = List[List[np.ndarray]]
NestedArrayDictList = List[List[StimDictArray]]

TensorList = List[torch.Tensor]
DictTensorList = List[StimDictTensor]
NestedTensorList = List[List[torch.Tensor]]
NestedTensorDictList = List[List[StimDictTensor]]


def flatten_nested_list(data: List[List[Any]]):
    return list(chain.from_iterable(data))