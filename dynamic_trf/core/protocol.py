from typing import List, Any, Tuple, Protocol

from . import (
    NestedTensorList, NestedTensorDictList, Configuration,
)
from .model import (
    MixedTRF, 
)
from mtrf.model import TRF

class TrainStep(Protocol):
    
    def __call__(
        self,
        trf:TRF,
        trf_lrg:TRF,
        train_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
        val_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
        configs: Configuration,
        folder:str,
        seed = 42, 
    ) -> MixedTRF:
        return MixedTRF('cpu', None, None)


class Trainer(Protocol):

    train_step: TrainStep

    def run(
        self,
        control_stims:NestedTensorList, 
        target_stims:NestedTensorDictList, 
        modulation_stims:NestedTensorDictList,
        resps:NestedTensorList, 
        configs:Configuration,
        montage = None,
    ):
        pass





    