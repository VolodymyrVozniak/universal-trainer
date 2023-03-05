from typing import Union, Dict, Any

from optuna.samplers import GridSampler

from .abstract import _Tuner


class GridTuner(_Tuner):
    """
    A class used to tune parameters with grid search.
    Use GridSampler from optuna under the hood.
    Link for more info:
    https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html#optuna.samplers.GridSampler

    Attributes:
        `study` (optuna.study.Study): A study corresponds to an optimization
        task, i.e., a set of trials.
        `params` (dict): Special type of dict with parameters to tune.
        Must include keys 'model' (dictionary with all paramaters for
        defined model class that will be used for training), 'optimizer'
        (dictionary with defined 'lr' and 'weight_decay') and 'batch_size'.
    """
    def __init__(
        self,
        params: Dict[str, Any],
        storage: Union[None, str],
        study_name: Union[None, str] = None,
        direction: str = "maximize",
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
            - `('categorical', (<values>))`: will be used for
            `suggest_categorical` optuna method;
            - `('constant', value)`: `value` will be used as constant for all
            trials and will not be tuned.

            Examples:
            - `{'hidden_features': ('categorical', (64, 128, 256)}` (meaning
            take one value from `(64, 128, 256)`);
            - `{'lr': ('constant', 1e-3}` (meaning always take `1e-3` value).

            `storage` (str): Database URL. If this argument is set to `None`,
            in-memory storage is used, and the Study will not be persistent.

            `study_name` (str): Study's name. If this argument is set to None,
            a unique name is generated automatically. Default is `None`.

            `direction` (str): Direction of optimization. Set 'minimize'
            for minimization and 'maximize' for maximization.
            Default is `'maximize'`.

            `load_if_exists` (bool): Flag to control the behavior to handle a
            conflict of study names. Default is `False`.
        """
        search_space = self.__get_search_space(params)
        sampler = GridSampler(search_space)
        super().__init__(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            params=params
        )

    @staticmethod
    def __get_search_space(params: Dict[str, Any]):
        search_space = {}

        def check_param_type(p_type):
            if p_type not in ["constant", "categorical"]:
                raise ValueError(f"{p_type} is not supported for GridTuner! "
                                 "Choose between 'constant' or 'categorical'!")

        for global_key in ["model", "optimizer"]:
            for param, (param_type, values) in params[global_key].items():
                check_param_type(param_type)
                if param_type == "categorical":
                    search_space[param] = list(values)

        param_type, values = params["batch_size"]
        check_param_type(param_type)
        if param_type == "categorical":
            search_space["batch_size"] = list(values)

        return search_space
