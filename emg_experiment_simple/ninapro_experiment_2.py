import datetime
from enum import Enum
import itertools
import logging
import os
import string
import warnings

from results_storage.results_storage import ResultsStorage
from sklearn.tree import DecisionTreeClassifier

from emg_experiment_simple.progressparallel import ProgressParallel
from emg_experiment_simple.stats_tools import (
    p_val_matrix_to_vec,
    p_val_vec_to_matrix,
)
from sklearn.naive_bayes import GaussianNB
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from statsmodels.stats.multitest import multipletests

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_channel_idx import (
    RawSignalsFilterChannelIdx,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    read_signals_from_dirs,
)


from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score

from joblib import delayed


from scipy.stats import wilcoxon
import seaborn as sns


import random
from scipy.stats import rankdata

# Plot line colors and markers
from cycler import cycler

from emg_experiment_simple import settings
from emg_experiment_simple.tools import logger
from emg_experiment_simple.xgb_classifier_label_enc import XGBClassifierWithLabelEncoder


N_INTERNAL_SPLITS = 4


class PlotConfigurer:

    def __init__(self) -> None:
        self.is_configured = False

    def configure_plots(self):
        if not self.is_configured:
            # print("Configuring plot")
            dcc = plt.rcParams["axes.prop_cycle"]
            mcc = cycler(
                marker=[
                    "o",
                    "s",
                    "D",
                    "^",
                    "v",
                    ">",
                    "<",
                    "p",
                    "*",
                    "x",
                    # "h",
                    # "H",
                    # "|",
                    # "_",
                ]
            )
            cc = cycler(
                color=[
                    "r",
                    "g",
                ]
            )

            lcc = cycler(
                linestyle=[
                    "-",
                    "--",
                    "-.",
                    ":",  # Default styles
                    (0, (1, 1)),  # Densely dotted
                    (0, (3, 1, 1, 1)),  # Short dash-dot
                    (0, (5, 1)),  # Loosely dashed
                    (0, (5, 5)),  # Medium dashed
                    (0, (8, 3, 2, 3)),  # Dash-dot-dot
                    (0, (10, 2, 2, 2)),  # Long dash, short dot
                    (0, (15, 5, 5, 2)),  # Complex pattern
                ]
            )
            c = lcc * (dcc + mcc)

            plt.rc("axes", prop_cycle=c)
            # print('Params set', plt.rcParams['axes.prop_cycle'])
            self.is_configured = True


configurer = PlotConfigurer()


def wavelet_extractor2(wavelet_level=2):
    extractor = SetCreatorDWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
            NpSignalExtractorMav(),
            NpSignalExtractorSsc(),
        ],
    )
    return extractor


def create_extractors():

    extractors_dict = {
        "DWT": wavelet_extractor2(),
    }

    return extractors_dict


def warn_unknown_labels(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    diffset = pred_set.difference(true_set)
    if len(diffset) > 0:
        warnings.warn("Diffset: {}".format(diffset))


def acc_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return accuracy_score(y_true, y_pred)


def bac_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return balanced_accuracy_score(y_true, y_pred)


def kappa_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return cohen_kappa_score(y_true, y_pred)


def f1_score_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return f1_score(y_true, y_pred, average="micro")


def precision_score_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return precision_score(y_true, y_pred, average="micro")


def recall_score_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return recall_score(y_true, y_pred, average="micro")


def generate_metrics():
    metrics = {
        "ACC": acc_m,
        "BAC": bac_m,
        "Kappa": kappa_m,
        "F1": f1_score_m,
    }
    return metrics


NUM_INNER_CV = 4


def generate_naive_bayes():
    return GaussianNB()


def generate_decision_tree():
    return DecisionTreeClassifier()


def generate_random_forest_t():
    classifier_dict = {
        "classifier_object": RandomForestClassifier(n_estimators=20, random_state=0),
        "params": {
            "max_depth": [5],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": [0.2],
            "max_samples": [0.1, 0.2, 0.4],
        },
    }

    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(balanced_accuracy_score)
    clf = GridSearchCV(
        classifier_object,
        classifier_params,
        cv=skf,
        scoring=bac_scorer,
        return_train_score=True,
        refit=True,
        verbose=10,
        error_score="raise",
    )
    return clf


def generate_xgboost_t():
    classifier_dict = {
        "classifier_object": XGBClassifierWithLabelEncoder(
            n_estimators=20, random_state=0
        ),
        "params": {
            "n_estimators": [20],
            "max_depth": [3],
            "learning_rate": np.linspace(0.05, 0.2, 3),
            "subsample": np.linspace(0.4, 0.8, 3),
            "colsample_bytree": np.linspace(0.1, 0.3, 3),
            "reg_alpha": [0.1],
            "reg_lambda": [0.1],
            "gamma": [0.05],
        },
    }
    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(balanced_accuracy_score)
    clf = GridSearchCV(
        classifier_object,
        classifier_params,
        cv=skf,
        scoring=bac_scorer,
        return_train_score=True,
        refit=True,
        verbose=10,
        error_score="raise",
    )
    return clf


# TODO uncomment
def generate_methods():
    methods = {
        "NB": generate_naive_bayes(),
        "DT": generate_decision_tree(),
        "RF": generate_random_forest_t(),
        "XG": generate_xgboost_t(),
    }
    return methods


class Dims(Enum):
    FOLDS = "folds"
    METHODS = "methods"


def run_experiment(
    datasets,
    output_directory,
    random_state=0,
    n_jobs=1,
    overwrite=True,
    n_channels=None,
    append=True,
    progress_log_handler=None,
    comment_str="",
):

    os.makedirs(output_directory, exist_ok=True)

    comment_file = os.path.join(output_directory, "comment.txt")
    with open(comment_file, "w") as f:
        f.write(comment_str)
        f.write("Start time: {}\n".format(datetime.datetime.now()))
        f.write("\n")
        f.write("Function Parameters:\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"n_jobs: {n_jobs}\n")
        f.write(f"overwrite: {overwrite}\n")
        f.write(f"n_channels: {n_channels}\n")
        f.write(f"append: {append}\n")
        f.write("\n")

    metrics = generate_metrics()
    n_metrics = len(metrics)

    extractors_dict = create_extractors()
    n_extr = len(extractors_dict)

    methods = generate_methods()
    n_methods = len(methods)

    for experiment_name, input_data_dir_list in datasets:

        logging.debug(f"Experiment: {experiment_name}")

        result_file_path = os.path.join(
            output_directory, "{}.csv".format(experiment_name)
        )
        exists = os.path.isfile(result_file_path)

        if exists and not (overwrite):
            print("Skipping {} !".format(experiment_name))
            continue

        logging.debug(f"Loading data for experiment {experiment_name}")
        raw_datasets = [
            read_signals_from_dirs(in_dir)["accepted"]
            for in_dir in input_data_dir_list
            if os.path.exists(in_dir)
        ]
        n_raw_datasets = len(raw_datasets)
        logging.debug(
            f"Loaded {n_raw_datasets} datasets for experiment {experiment_name}"
        )
        if n_raw_datasets == 0:
            logging.warning(f"No data for experiment {experiment_name}")
            continue

        logging.debug("Filtering")
        if n_channels is not None:
            for raw_set_idx, raw_set in enumerate(raw_datasets):
                n_set_channels = raw_set[0].to_numpy().shape[1]
                n_effective_channels = min((n_set_channels, n_channels))
                indices = [*range(n_effective_channels)]
                filter = RawSignalsFilterChannelIdx(indices)
                raw_datasets[raw_set_idx] = filter.fit_transform(raw_set)

        logging.debug("Filtering done")

        extractor = wavelet_extractor2()

        extr_datasets_X = []
        extr_datasets_y = []
        groups = []
        for raw_set_idx, raw_set in enumerate(raw_datasets):
            X, y, z = extractor.fit_transform(raw_set)
            extr_datasets_X.append(X)
            extr_datasets_y.append(y)
            groups += [raw_set_idx] * len(y)

        skf = LeaveOneGroupOut()
        X_all = np.vstack(extr_datasets_X)
        y_all = np.concatenate(extr_datasets_y)

        def compute(fold_idx, train_idx, test_idx):

            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            fold_res = []

            for method_name in tqdm(
                methods,
                leave=False,
                total=n_methods,
                desc="Methods, Fold {}".format(fold_idx),
            ):

                method = methods[method_name]

                method.fit(X_train, y_train)
                y_pred = method.predict(X_test)

                fold_res.append(
                    (
                        method_name,
                        fold_idx,
                        y_test,
                        y_pred
                    )
                )
            return fold_res

        results_list = ProgressParallel(
            n_jobs=n_jobs,
            desc=f"K-folds for experiment: {experiment_name}",
            total=n_raw_datasets,
            leave=False,
            file_handler=progress_log_handler,
        )(
            delayed(compute)(fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(X_all, y_all, groups=groups)
            )
        )

        records = []
        for result_sublist in results_list:
            for (
                method_name,
                fold_idx,
                y_test,
                y_pred,
            ) in result_sublist:
                for y_idx, _ in enumerate(y_test):
                    records.append(
                        {
                            Dims.METHODS.value:method_name,
                            Dims.FOLDS.value:fold_idx,
                            "y_test":y_test[y_idx],
                            "y_pred":y_pred[y_idx],
                        }
                    )

        res_df = pd.DataFrame(records)
        logging.debug("Dumping results")
        with open(result_file_path, "wb") as fh:
            res_df.to_csv(fh)


def analyze_results(results_directory, output_directory, alpha=0.05):

    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".csv")]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        results_df = pd.read_csv(result_file_path)
        results_df.head(10)

        cm_file_path = os.path.join(output_directory,f"{result_file_basename}_cm.md")
        
        with open(cm_file_path,"w") as cm_file_handler:
            print("# Method specific analysis\n", file=cm_file_handler)
            
            overall_df = pd.DataFrame()
            for method_name, g in results_df.groupby(Dims.METHODS.value):
                print(f"## {method_name}\n", file=cm_file_handler)
                print(f"### Confusion matrix\n", file=cm_file_handler)
                cm = confusion_matrix(g["y_test"], g["y_pred"])
                u_labels = np.unique( np.hstack((g["y_test"],g["y_pred"])))
                cm_df = pd.DataFrame(cm, index=u_labels, columns=u_labels)
                cm_df.to_markdown(cm_file_handler)
                print("\n", file=cm_file_handler)

                print(f"### classification report\n", file=cm_file_handler)
                cr_dict = classification_report(g["y_test"],g["y_pred"],output_dict=True)
                cr_df = pd.DataFrame(cr_dict).transpose()
                class_rows = cr_df.iloc[:-3]  # assumes last 3 rows are avg metrics

                overall_df[f"{method_name}-f1"] = class_rows["f1-score"]
                class_rows_sorted = class_rows.sort_values("f1-score", ascending=False)
                
                class_rows_sorted.to_markdown(cm_file_handler)
                print("\n", file=cm_file_handler)

            print("# Overall ranks", file=cm_file_handler)
            ranked_df = rankdata(overall_df, axis=0, method="average")
            av_ranks_df = pd.DataFrame(ranked_df.mean(axis=1),index=overall_df.index,columns=["avg-rank"])
            av_ranks_sorted = av_ranks_df.sort_values("avg-rank",ascending=False)
            av_ranks_sorted.to_markdown(cm_file_handler)
            print("\n", file=cm_file_handler)
            



def main():
    np.random.seed(0)
    random.seed(0)

    subjects = list([*range(1, 12)])
    experiments = list([*range(1, 4)])
    labels = ["stimulus", "restimulus"]

    db_name = "db3"

    data_sets = []
    for experiment in experiments:
        for label in labels:
            data_sets.append(
                (
                    f"exp_{experiment}_{label}",
                    [
                        os.path.join(
                            settings.DATAPATH, db_name, f"S{su}_E{experiment}_A1_{label}"
                        )
                        for su in subjects
                    ],
                )
            )

    output_directory = os.path.join(
        settings.EXPERIMENTS_RESULTS_PATH,
        f"./ninapro_{db_name}_experiment/",
    )
    os.makedirs(output_directory, exist_ok=True)

    log_dir = settings.EXPERIMENTS_LOGS_PATH
    log_file = os.path.splitext(os.path.basename(__file__))[0]
    logger(log_dir, log_file, enable_logging=True)

    progress_log_path = os.path.join(output_directory, "progress.log")
    progress_log_handler = open(progress_log_path, "w")

    comment_str = """
    Simple experiment.
    """
    run_experiment(
        data_sets,
        output_directory,
        random_state=0,
        n_jobs=-1,
        overwrite=True,
        n_channels=12,
        progress_log_handler=progress_log_handler,
        comment_str=comment_str,
    )

    analysis_functions = [
        analyze_results,
    ]
    alpha = 0.05

    ProgressParallel(
        backend="multiprocessing",
        n_jobs=None,
        desc="Analysis",
        total=len(analysis_functions),
        leave=False,
    )(
        delayed(fun)(output_directory, output_directory, alpha)
        for fun in analysis_functions
    )


if __name__ == "__main__":
    main()
