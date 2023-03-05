from typing import Union, Dict, Any

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
        params: Dict[str, Any]
    ):
        self.study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists
        )
        self.params = params
