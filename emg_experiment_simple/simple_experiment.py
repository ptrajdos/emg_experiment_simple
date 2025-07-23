import datetime
from enum import Enum
import logging
import os
import string
import warnings

from results_storage.results_storage import ResultsStorage

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
    cohen_kappa_score,
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
        "RF": generate_random_forest_t(),
        "XG": generate_xgboost_t(),
    }
    return methods


class Dims(Enum):
    FOLDS = "folds"
    METRICS = "metrics"
    EXTRACTORS = "extractors"
    METHODS = "methods"


def run_experiment(
    input_data_dir_list,
    output_directory,
    n_splits=10,
    n_repeats=3,
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
        f.write(f"n_splits: {n_splits}\n")
        f.write(f"n_repeats: {n_repeats}\n")
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

    skf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_folds = skf.get_n_splits()

    coords = {
        Dims.METRICS.value: [k for k in metrics],
        Dims.EXTRACTORS.value: [k for k in extractors_dict],
        Dims.METHODS.value: [k for k in methods],
        Dims.FOLDS.value: [k for k in range(n_folds)],
    }

    for in_dir in tqdm(input_data_dir_list, desc="Data sets"):

        set_name = os.path.basename(in_dir)
        result_file_path = os.path.join(output_directory, "{}.pickle".format(set_name))
        exists = os.path.isfile(result_file_path)

        if exists and not (overwrite):
            print("Skipping {} !".format(set_name))
            continue

        pre_set = read_signals_from_dirs(in_dir)
        raw_set = pre_set["accepted"]

        if n_channels is not None:
            n_set_channels = raw_set[0].to_numpy().shape[1]
            n_effective_channels = min((n_set_channels, n_channels))
            indices = [*range(n_effective_channels)]
            filter = RawSignalsFilterChannelIdx(indices)
            raw_set = filter.fit_transform(raw_set)

        y = np.asanyarray(raw_set.get_labels())
        num_labels = len(np.unique(y))

        results_storage = ResultsStorage.init_coords(coords=coords, name="Storage")
        if os.path.exists(result_file_path):
            with open(result_file_path, "rb") as fh:
                loaded_storage = pickle.load(fh)
            loaded_storage.name = "loaded"
            results_storage = ResultsStorage.merge_with_loaded(
                loaded_obj=loaded_storage, new_obj=results_storage
            )

        def compute(fold_idx, train_idx, test_idx):

            raw_train = raw_set[train_idx]
            raw_test = raw_set[test_idx]

            fold_res = []

            for extractor_name in ResultsStorage.coords_need_recalc(
                results_storage, Dims.EXTRACTORS.value
            ):
                extractor = extractors_dict[extractor_name]

                X_train, y_train, _ = extractor.fit_transform(raw_train)
                X_test, y_test, _ = extractor.transform(raw_test)

                for method_name in tqdm(
                    ResultsStorage.coords_need_recalc(
                        results_storage, Dims.METHODS.value
                    ),
                    leave=False,
                    total=n_methods,
                    desc="Methods, Fold {}".format(fold_idx),
                ):
                    
                    method = methods[method_name]

                    method.fit(X_train, y_train)
                    y_pred = method.predict(X_test)

                    for metric_name in ResultsStorage.coords_need_recalc(
                        results_storage, Dims.METRICS.value
                    ):
                        metric = metrics[metric_name]

                        metric_value = metric(y_test, y_pred)

                        # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
                        fold_res.append(
                            (
                                metric_name,
                                extractor_name,
                                method_name,
                                fold_idx,
                                metric_value,
                            )
                        )
            return fold_res

        results_list = ProgressParallel(
            n_jobs=n_jobs,
            desc=f"K-folds for set: {set_name}",
            total=skf.get_n_splits(),
            leave=False,
            file_handler=progress_log_handler,
        )(
            delayed(compute)(fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(raw_set, y))
        )

        for result_sublist in results_list:
            # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
            for (
                metric_name,
                extractor_name,
                method_name,
                fold_idx,
                metric_value,
            ) in result_sublist:
                results_storage.loc[
                    {
                        Dims.METRICS.value: metric_name,
                        Dims.EXTRACTORS.value: extractor_name,
                        Dims.METHODS.value: method_name,
                        Dims.FOLDS.value: fold_idx,
                    }
                ] = metric_value

        logging.debug("Dumping results")
        with open(result_file_path, "wb") as fh:
            pickle.dump(results_storage, file=fh)


def analyze_results(results_directory, output_directory, alpha=0.05):

    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle")]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        results_holder = pickle.load(open(result_file_path, "rb"))

        method_names = results_holder[Dims.METHODS.value].values
        n_methods = len(method_names)

        pdf_file_path = os.path.join(
            output_directory, "{}.pdf".format(result_file_basename)
        )
        report_file_path = os.path.join(
            output_directory, "{}.md".format(result_file_basename)
        )
        report_file_handler = open(report_file_path, "w")

        with PdfPages(pdf_file_path) as pdf:

            for metric_name in results_holder[Dims.METRICS.value].values:
                print("# {}".format(metric_name), file=report_file_handler)
                for extractor_name in results_holder[Dims.EXTRACTORS.value].values:
                    print("## {}".format(extractor_name), file=report_file_handler)
                    
                    # mehods x folds
                    sub_results = results_holder.loc[
                        {
                            Dims.METRICS.value: metric_name,
                            Dims.EXTRACTORS.value: extractor_name,
                        }
                    ].to_numpy()

                    plt.boxplot(sub_results.transpose())
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.title(
                        "{}, {}".format(
                            metric_name,
                            extractor_name,
                        )
                    )
                    plt.xticks(
                        range(1, len(method_names) + 1),
                        method_names,
                    )
                    pdf.savefig()
                    plt.close()

                    p_vals = np.zeros((n_methods, n_methods))
                    values = sub_results.transpose()

                    for i in range(n_methods):
                        for j in range(n_methods):
                            if i == j:
                                continue

                            values_squared_diff = np.sqrt(
                                np.sum(
                                    (values[:, i] - values[:, j])
                                    ** 2
                                )
                            )
                            if values_squared_diff > 1e-4:
                                with warnings.catch_warnings():  # Normal approximation
                                    warnings.simplefilter("ignore")
                                    p_vals[i, j] = wilcoxon(
                                        values[:, i],
                                        values[:, j],
                                    ).pvalue  # mannwhitneyu(values[:,i], values[:,j]).pvalue
                            else:
                                p_vals[i, j] = 1.0

                    p_val_vec = p_val_matrix_to_vec(p_vals)

                    p_val_vec_corrected = multipletests(
                        p_val_vec, method="hommel"
                    )

                    corr_p_val_matrix = p_val_vec_to_matrix(
                        p_val_vec_corrected[1], n_methods
                    )

                    p_val_df = pd.DataFrame(
                        corr_p_val_matrix,
                        columns=method_names,
                        index=method_names,
                    )
                    print(
                        "PVD:\n",
                        p_val_df,
                        file=report_file_handler,
                    )

        report_file_handler.close()

def main():
    np.random.seed(0)
    random.seed(0)

    data_path0B = os.path.join(settings.DATAPATH, "MK_10_03_2022_EMG")

    data_sets = [data_path0B]

    output_directory = os.path.join(
        settings.EXPERIMENTS_RESULTS_PATH,
        "./results_simple_experiment/",
    )
    os.makedirs(output_directory, exist_ok=True)

    log_dir = os.path.dirname(settings.EXPERIMENTS_LOGS_PATH)
    log_file = os.path.splitext(os.path.basename(__file__))[0]
    logger(log_dir, log_file, enable_logging=False)

    progress_log_path = os.path.join(output_directory, "progress.log")
    progress_log_handler = open(progress_log_path, "w")

    comment_str = """
    Simple experiment.
    """
    run_experiment(
        data_sets,
        output_directory,
        n_splits=10,  
        n_repeats=4, 
        random_state=0,
        n_jobs=-1,
        overwrite=True,
        n_channels=8,
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
