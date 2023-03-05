from typing import Union, Dict, Any

from optuna.samplers import RandomSampler

from .abstract import _Tuner


class RandomTuner(_Tuner):
    """
    A class used to tune parameters with random sampling.
    Use RandomSampler from optuna under the hood.
    Link for more info:
    https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler

    Attributes:
        `study` (optuna.study.Study): A study corresponds to an optimization
        task, i.e., a set of trials.
        `params` (dict): Special type of dict with parameters to tune.
        Must include keys 'model' (dictionary with all paramaters for
        defined model class that will be used for training), 'optimizer'
        (dectionary with defined 'lr' and 'weight_decay') and 'batch_size'.
    """
    def __init__(
        self,
        params: Dict[str, Any],
        storage: Union[None, str],
        study_name: Union[None, str] = None,
        direction: str = "minimize",
        load_if_exists: bool = False
    ):
        """
        Args:
            `params` (dict): Special type of dict with parameters to tune.

            Must include keys:
            - 'model' (dictionary with all paramaters for
            defined model class that will be used for training);
            - 'optimizer' (dictionary with defined 'lr' and
            'weight_decay');
            - 'batch_size'.

            All entries for any parameter must be defined in the following way:
            {'`param`': (`param_type`, (`values`))}.
            Possible `param_type` and `values`:
            - `('int', (low, high, step, log))` (will be used for `suggest_int`
            optuna method; `low` and `high` are both included in the range);
            - `('float', (low, high, log))` (will be used for `suggest_float`
            optuna method; `low` and `high` are both included in the range);
            - `('categorical', (<values>))` (will be used for
            `suggest_categorical` optuna method;
            - `('constant', value)` (`value` will be used as constant for all
            trials and will not be tuned).

            Examples:
            - `{'n_layers': ('int', (1, 3, 1, False))}` (meaning take int
            value from `[1, 3]` range with `step=1` and `log=False`);
            - `{'dropout': ('float', (0.1, 0.5, False))}` (meaning take
            float value from `[0.1, 0.5]` range with `log=False`);
            - `{'hidden_features': ('categorical', (64, 128, 256)}` (meaning
            take one value from `(64, 128, 256)`);
            - `{'lr': ('constant', 1e-3}` (meaning always take `1e-3` value).

            `storage` (str): Database URL. If this argument is set to `None`,
            in-memory storage is used, and the Study will not be persistent.

            `study_name` (str): Study's name. If this argument is set to None,
            a unique name is generated automatically. Default is `None`.

            `direction` (str): Direction of optimization. Set 'minimize'
            for minimization and 'maximize' for maximization.
            Default is `'maximize'`:

            `load_if_exists` (bool): Flag to control the behavior to handle a
            conflict of study names. Default is `False`.
        """
        sampler = RandomSampler()
        super().__init__(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            params=params
        )
