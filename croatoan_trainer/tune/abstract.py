from typing import Union, Dict, Any, Callable, Tuple

import optuna
from optuna.samplers import BaseSampler


class _Tuner():
    def __init__(
        self,
        storage: Union[None, str],
        sampler: BaseSampler,
        study_name: Union[None, str],
        direction: str,
        load_if_exists: bool,
        params: Union[Dict[str, Any], Callable[
            [optuna.trial.Trial], Tuple[Dict[str, Any], Dict[str, Any], int]
        ]],
    ):
        self.study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists
        )
        self.params = params
