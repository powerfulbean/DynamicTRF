
from typing import List, Dict, TypeVar, Generic, Any, Annotated, Tuple, TypedDict
# from typing_extensions  import TypedDict as OldTypedDict, TypeVar as OldTypeVar

from dataclasses import dataclass
from itertools import chain

import torch
import numpy as np

from ..utils.io import checkFolder

"""
{
    'x': discrete_values,
    'timeinfo': [2, same_length_of_x] 2: wordonset and wordoffset
}
"""

class StimDictTensor(TypedDict):
    x: torch.Tensor
    timeinfo: torch.Tensor

@dataclass
class Configuration:
    folderName: str = 'dynamic_trf'
    workspace: str = './'
    mtrf_only: bool = False
    contextModel: str = 'CausalConv'
    nContextWin: int = 2
    fTRFMode: str = '+-a,b'
    nBasis: int = 21
    timelag: Tuple[int, int] = (0, 700)
    nFolds: int = 10
    epoch: int = 100
    batchSize: int = 1
    wd: float = 0.01
    lr: Tuple[float, float]= (0.001,0.001)
    optimizer: str = 'AdamW'
    lrScheduler: str = 'cycle'
    randomSeed: int = 42
    fs: int = -1
    extraTimeLag: int = 200
    device:str = 'cpu'
    checkpoint:bool = False
    lambda_range_power: Tuple[float, float] = (-4, 4)

    @property
    def tarDir(self):
        out = self.workspace + '/' + self.folderName
        checkFolder(out)
        return out
    
    @property
    def limitOfShift_idx(self):
        return int(np.ceil(self.fs * self.extraTimeLag/1000))

ScalarTensor = Annotated[torch.Tensor, "scalar (shape=())"]

# StimDictArray = StimDict[np.ndarray]
# StimDictTensor = StimDict[torch.Tensor]


NestedArrayList = List[List[np.ndarray]]
# NestedArrayDictList = List[List[StimDictArray]]

TensorList = List[torch.Tensor]
DictTensorList = List[StimDictTensor]
NestedTensorList = List[List[torch.Tensor]]
NestedTensorDictList = List[List[StimDictTensor]]


def flatten_nested_list(data: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(data))
