from typing import List, Dict, Union, Any


class ResultsConverter():
    """
    A class used to convert classic ML results to a dictionary
    that can be passed to analyzer classes.

    Attributes:
        `ids` (list):
            List with unique ids.
        `true` (list):
            List with real values.
        `pred` (list):
            List with predicted values.
        `metrics` (dict):
            Dictionary with metric names as keys and metric scores as values.

    Methods:
        `get_results(stage)`:
            Gets results for analyzer classes.
    """

    def __init__(
        self,
        ids: List[Union[str, int]],
        true: List[float],
        pred: List[float],
        metrics: Dict[str, float]
    ):
        """
        Args:
            `ids` (list):
                List with unique ids.
            `true` (list):
                List with real values.
            `pred` (list):
                List with predicted values.
            `metrics` (dict):
                Dictionary with metric names as keys
                and metric scores as values.
        """
        self.ids = ids
        self.pred = pred
        self.true = true
        self.pred = pred
        self.metrics = metrics

    def get_results(self, stage: str) -> Dict[str, Dict[str, Any]]:
        """
        Gets results for analyzer classes.

        Args:
            `stage` (str):
                Stage to use as key in resulting dict.

        Returns:
            dict: Dictionary for analyzer classes.
        """
        return {stage: {
            "best_result": {"metrics": self.metrics, "epoch": 0},
            "ids": self.ids,
            "true": self.true,
            "pred": self.pred
        }}
