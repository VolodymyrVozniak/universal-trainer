from typing import Union, Dict, Any, Callable, Tuple

import optuna
from optuna.samplers import TPESampler

from .abstract import _Tuner


class TPETuner(_Tuner):
    """
    A class used to tune parameters with TPE (Tree-structured Parzen
    Estimator) algorithm. Use TPRSampler from optuna under the hood.
    Link for more info:
    https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler

    Attributes:
        `study` (optuna.study.Study):
            A study corresponds to an optimization task, i.e., a set of trials.
        `params` (dict):
            Special type of dict with parameters to tune.
            Must include keys 'model' (dictionary with all paramaters for
            defined model class that will be used for training), 'optimizer'
            (dictionary with defined 'lr' and 'weight_decay') and 'batch_size'.
    """
    def __init__(
        self,
        params: Union[Dict[str, Any], Callable[
            [optuna.trial.Trial], Tuple[Dict[str, Any], Dict[str, Any], int]
        ]],
        storage: Union[None, str],
        study_name: Union[None, str] = None,
        direction: str = "minimize",
        load_if_exists: bool = False
    ):
        """
        Args:
            `params` (dict):
                Special type of dict with parameters to tune.

                Must include keys:
                - 'model' (dictionary with all paramaters for
                defined model class that will be used for training);
                - 'optimizer' (dictionary with defined 'lr' and
                'weight_decay');
                - 'batch_size'.

                All entries for any parameter must be defined in following way:
                {'`param`': (`param_type`, (`values`))}.
                Possible `param_type` and `values`:
                - `('int', (low, high, step, log))`: will be used for
                `suggest_int` optuna method; `low` and `high` are both
                included in the range;
                - `('float', (low, high, log))`: will be used for
                `suggest_float` optuna method; `low` and `high` are both
                included in the range;
                - `('categorical', (<values>))`: will be used for
                `suggest_categorical` optuna method;
                - `('constant', value)`: `value` will be used as constant for
                all trials and will not be tuned.

                Examples:
                - `{'n_layers': ('int', (1, 3, 1, False))}` (meaning take int
                value from `[1, 3]` range with `step=1` and `log=False`);
                - `{'dropout': ('float', (0.1, 0.5, False))}` (meaning take
                float value from `[0.1, 0.5]` range with `log=False`);
                - `{'hidden_features': ('categorical', (64, 128, 256)}`
                (meaning take one value from `(64, 128, 256)`);
                - `{'lr': ('constant', 1e-3}`
                (meaning always take `1e-3` value).

            `params` (callable):
                Special function that defines parameters for tuning using
                optuna trial. It takes optuna trial as input, defines
                parameters and gives model parameters and optimizer parameters
                as dict outputs plus batch size as integer. Example:
                ```
                def get_tune_params(trial: optuna.trial.Trial):
                    model_params, optimizer_params = {}, {}

                    model_params['in_features'] = 512  # some constant value

                    model_params['activation'] = trial.suggest_categorical(
                        'activation', ['ReLU', 'GELU', 'ELU', 'LeakyReLU']
                    )

                    n_layers = trial.suggest_int('n_layers', 2, 4)
                    model_params['n_layers'] = n_layers

                    for i in range(n_layers):
                        n_units = trial.suggest_categorical(
                            f'n_units_l{i}', (512, 1024, 2048)
                        )
                        dropout = trial.suggest_float(
                            f'dropout_l{i}', 0.1, 0.5
                        )
                        model_params[f'n_units_l{i}'] = n_units
                        model_params[f'dropout_l{i}'] = dropout

                    optimizer_params['lr'] = trial.suggest_float(
                        'lr', 1e-5, 1e-1, log=True
                    )
                    optimizer_params['weight_decay'] = trial.suggest_float(
                        'weight_decay', 5e-5, 5e-3, log=True
                    )

                    batch_size = trial.suggest_categorical(
                        "batch_size", (16, 32, 64, 128, 256, 512, 1024, 2048)
                    )

                    return model_params, optimizer_params, batch_size
                ```

            `storage` (str):
                Database URL. If this argument is set to `None`, in-memory
                storage is used, and the Study will not be persistent.

            `study_name` (str):
                Study's name. If this argument is set to None,
                a unique name is generated automatically. Default is `None`.

            `direction` (str):
                Direction of optimization. Set 'minimize' for minimization and
                'maximize' for maximization. Default is `'maximize'`.

            `load_if_exists` (bool):
                Flag to control the behavior to handle a conflict of
                study names. Default is `False`.
        """
        sampler = TPESampler()
        super().__init__(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            params=params
        )
