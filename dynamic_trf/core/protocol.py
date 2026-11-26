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
        mixed_rf: MixedTRF,
        train_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
        val_data:Tuple[List[Any], List[Any], List[Any], List[Any]],
        configs: Configuration,
        folder:str,
    ) -> MixedTRF:
        return MixedTRF('cpu', None, None)


class Trainer(Protocol):

    train_step: TrainStep

    def prepare_mixedtrf(
            self,
            trainerDir,
            trf:TRF,
            trf_lrg:TRF,
            train_data,
            configs: Configuration,
            seed = 42,
        ) -> MixedTRF:
        return MixedTRF(None, None, None)

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





    