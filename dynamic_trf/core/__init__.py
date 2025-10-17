
from typing import List, Dict, TypeVar, Generic, Any
from dataclasses import dataclass, asdict
from itertools import chain

import torch
import numpy as np

Array = TypeVar('Array')
@dataclass
class StimDict(Generic[Array]):
    x: Array
    timeinfo: Array

@dataclass
class Configuration:
    folderName: str = 'dynamic_trf'
    tarDir: str = './'
    mtrf_only: bool = False
    contextModel: str = 'CausalConv'
    nContextWin: int = 2
    fTRFMode: str = '+-a,b'
    nBasis: int = 21
    timelag: List[int] = [0, 700]
    nFolds: int = 10
    epoch: int = 100
    batchSize: int = 1
    wd: float = 0.01
    lr: list[float, float]= [0.001,0.001]
    optimizer: str = 'AdamW'
    lrScheduler: str = 'cycle'
    randomSeed: int = 42



StimDictArray = StimDict[np.ndarray]
StimDictTensor = StimDict[torch.Tensor]


NestedArrayList = List[List[np.ndarray]]
NestedArrayDictList = List[List[StimDictArray]]

NestedTensorList = List[List[torch.Tensor]]
NestedTensorDictList = List[List[StimDictTensor]]


def flatten_nested_list(data: List[List[Any]]):
    return list(chain.from_iterable(data))