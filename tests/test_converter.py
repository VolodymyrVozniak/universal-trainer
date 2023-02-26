import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score

from croatoan_trainer.analyze import BinaryAnalyzer
from croatoan_trainer.utils import ResultsConverter


def test_converter():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LinearSVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    def get_metrics(y_true, y_pred):
        scores = {}
        scores["accuracy"] = float(accuracy_score(y_true, y_pred))
        scores["recall"] = float(recall_score(y_true, y_pred))
        scores["precision"] = float(precision_score(y_true, y_pred))
        scores["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        return scores

    ids = np.arange(len(y_test))
    metrics = get_metrics(y_test, y_pred)

    converter = ResultsConverter(ids, y_test, y_pred, metrics)
    results = converter.get_results("test")

    analyzer = BinaryAnalyzer(results)

    analyzer.get_stages()
    analyzer.get_metrics()
    analyzer.get_df_pred("test")

    analyzer.get_df_metrics(["test"])
    analyzer.get_metric_result("test", jaccard_score, True, zero_division=0)

    analyzer.print_classification_report("test")

    analyzer.plot_confusion_matrix("test")
    analyzer.plot_pred_hist("test")
    analyzer.plot_roc_auc("test")
    analyzer.plot_precision_recall_auc("test")
    analyzer.plot_enrichment("test")
