from copy import deepcopy
import datetime
from enum import Enum
import logging
import os
import warnings

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_kurtosis import (
    NpSignalExtractorKurtosis,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_skew import (
    NpSignalExtractorSkew,
)
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from emg_experiment_simple.progressparallel import ProgressParallel
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

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_swt import (
    SetCreatorSWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_channel_idx import (
    RawSignalsFilterChannelIdx,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_var import (
    NpSignalExtractorVar,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_rms import (
    NpSignalExtractorRms,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mobility import (
    NpSignalExtractorMobility,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_complexity import (
    NpSignalExtractorComplexity,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_higuchi_fd2 import (
    NpSignalExtractorHiguchiFD2
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_moment import (
    NpSignalExtractorTemporalMoment,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_kurtosis import (
    NpSignalExtractorTemporalKurtosis
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_skew import (
    NpSignalExtractorTemporalSkew
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import NpSignalExtractorSpectralMoment
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_skew import NpSignalExtractorSpectralSkew
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_kurtosis import NpSignalExtractorSpectralKurtosis
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    read_signals_from_archive,
)


from tqdm import tqdm
from sklearn.metrics import accuracy_score

from joblib import delayed


import seaborn as sns


import random
from scipy.stats import rankdata

# Plot line colors and markers
from cycler import cycler

from emg_experiment_simple import settings
from emg_experiment_simple.tools import logger
from emg_experiment_simple.xgb_classifier_label_enc import XGBClassifierWithLabelEncoder

from matplotlib.backends.backend_pdf import PdfPages
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import (
    RawSignalsFilterAllPass,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_robuts_standarizer import (
    RawSignalsFilterAllRobustStandarizer,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_standarizer import (
    RawSignalsFilterAllStandarizer,
)


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


def wavelet_extractor2_DWT(wavelet_level=3):
    extractor = SetCreatorDWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
            NpSignalExtractorVar(),
            NpSignalExtractorKurtosis(),
            NpSignalExtractorMobility(),
            NpSignalExtractorComplexity(),
            NpSignalExtractorHiguchiFD2(),
            NpSignalExtractorTemporalMoment(order=1),
            NpSignalExtractorTemporalMoment(order=2),
            NpSignalExtractorTemporalSkew(),
            NpSignalExtractorTemporalKurtosis(),
        ]
    )
    return extractor

def wavelet_extractor2_SWT(wavelet_level=3):
    extractor = SetCreatorSWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
            NpSignalExtractorVar(),
            NpSignalExtractorKurtosis(),
            NpSignalExtractorMobility(),
            NpSignalExtractorComplexity(),
            NpSignalExtractorHiguchiFD2(),
            NpSignalExtractorTemporalMoment(order=1),
            NpSignalExtractorTemporalMoment(order=2),
            NpSignalExtractorTemporalSkew(),
            NpSignalExtractorTemporalKurtosis(),
            NpSignalExtractorSpectralMoment(order=1),
            NpSignalExtractorSpectralMoment(order=2),
            NpSignalExtractorSpectralSkew(),
            NpSignalExtractorSpectralKurtosis(),
        ]
    )
    return extractor

def create_extractors():

    extractors_dict = {
        # "DWT": wavelet_extractor2_DWT(wavelet_level=3),
        "SWT": wavelet_extractor2_SWT(wavelet_level=3),
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


def geometric_mean_score_m(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred, average="micro", pos_label=None)


def generate_metrics():
    metrics = {
        "ACC": acc_m,
        "BAC": bac_m,
        "Kappa": kappa_m,
        "F1": f1_score_m,
    }
    return metrics


NUM_INNER_CV = 5


def generate_naive_bayes():
    return GaussianNB()


def generate_decision_tree():
    return DecisionTreeClassifier()


def generate_random_forest_t():
    classifier_dict = {
        "classifier_object": RandomForestClassifier(n_estimators=100, random_state=0),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "max_samples": [0.1, 0.2, 0.4, 0.5, None],
        },
    }

    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
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


def generate_knn_t():
    classifier_dict = {
        "classifier_object": KNeighborsClassifier(),
        "params": {
            "n_neighbors": list(range(1, 27, 2)),  # ATTENTION to 27
        },
    }

    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
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


def generate_SVC_linear_t():
    classifier_dict = {
        "classifier_object": SVC(
            kernel="linear", random_state=0, decision_function_shape="ovo"
        ),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
    }

    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
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


def generate_SVC_rbf_t():
    classifier_dict = {
        "classifier_object": SVC(
            kernel="rbf", random_state=0, decision_function_shape="ovo"
        ),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100],
        },
    }

    classifier_object = classifier_dict["classifier_object"]
    classifier_params = classifier_dict["params"]

    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
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
            n_estimators=100, random_state=0
        ),
        "params": {
            "n_estimators": [100],
            "max_depth": [3, 5, None],
            "learning_rate": np.linspace(0.05, 0.5, 5),
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
    bac_scorer = make_scorer(kappa_m)
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


def generate_ecoc_xgb_t():

    params = {
        "estimator__code_size": [2, 3],
        "estimator__estimator__max_depth": [3, 5, None],
        "estimator__estimator__learning_rate": np.linspace(0.05, 0.2, 3),
        "estimator__estimator__subsample": np.linspace(0.4, 0.8, 3),
        "estimator__estimator__colsample_bytree": np.linspace(0.1, 0.3, 3),
        "estimator__estimator__reg_alpha": [0.1],
        "estimator__estimator__reg_lambda": [0.1],
        "estimator__estimator__gamma": [0.05],
    }

    pipeline = Pipeline(
        [
            (
                "estimator",
                OutputCodeClassifier(
                    estimator=XGBClassifierWithLabelEncoder(
                        n_estimators=20, random_state=0
                    ),
                    random_state=0,
                ),
            )
        ]
    )
    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=bac_scorer,
        cv=skf,
        return_train_score=True,
        refit=True,
        verbose=10,
        error_score="raise",
    )
    return gs


def generate_ecoc_rf_t():

    params = {
        "estimator__code_size": [2, 3],
        "estimator__estimator__max_depth": [3, 5, 10, None],
        "estimator__estimator__min_samples_split": [2, 5, 10],
        "estimator__estimator__min_samples_leaf": [1, 2, 4],
        "estimator__estimator__max_features": ["sqrt", "log2", None],
        "estimator__estimator__max_samples": [0.1, 0.2, 0.4, 0.5, None],
    }

    pipeline = Pipeline(
        [
            (
                "estimator",
                OutputCodeClassifier(
                    estimator=RandomForestClassifier(n_estimators=100, random_state=0),
                    random_state=0,
                ),
            )
        ]
    )
    skf = RepeatedStratifiedKFold(n_splits=NUM_INNER_CV, n_repeats=1, random_state=0)
    bac_scorer = make_scorer(kappa_m)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=bac_scorer,
        cv=skf,
        return_train_score=True,
        refit=True,
        verbose=10,
        error_score="raise",
    )
    return gs


# TODO uncomment
def generate_methods():
    methods = {
        "NB": generate_naive_bayes(),
        "DT": generate_decision_tree(),
        "RF": generate_random_forest_t(),
        "XG": generate_xgboost_t(),
        # "ECOC-XG": generate_ecoc_xgb_t(),
        # "ECOC-RF": generate_ecoc_rf_t(),
        "KNN": generate_knn_t(),
        # "SVC-linear": generate_SVC_linear_t(),
        # "SVC-rbf": generate_SVC_rbf_t(),
    }
    return methods


def generate_filters():
    filters = {
        # "NoFilter": RawSignalsFilterAllPass(),
        "Stand": RawSignalsFilterAllStandarizer(),
        # "RobustStand":RawSignalsFilterAllRobustStandarizer(),
    }
    return filters


class Dims(Enum):
    FOLDS = "folds"
    METHODS = "methods"


def run_experiment(
    datasets,
    output_directory,
    random_state=0,
    n_jobs=-1,
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

    for experiment_name, archive_path, selected_classes, input_data_regexes in datasets:

        logging.debug(f"Experiment: {experiment_name}")

        result_file_path = os.path.join(
            output_directory, "{}.csv".format(experiment_name)
        )
        exists = os.path.isfile(result_file_path)

        if exists and not (overwrite):
            print("Skipping {} !".format(experiment_name))
            continue

        logging.debug(f"Loading data for experiment {experiment_name}")
        raw_datasets = list()
        for input_regex in input_data_regexes:
            raw_data = read_signals_from_archive(
                archive_path, filter_regex=input_regex
            )["accepted"]
            if len(raw_data) > 0:
                raw_datasets.append(raw_data)

        n_raw_datasets = len(raw_datasets)
        logging.debug(
            f"Loaded {n_raw_datasets} datasets for experiment {experiment_name}"
        )
        if n_raw_datasets == 0:
            logging.warning(f"No data for experiment {experiment_name}")
            continue

        logging.debug("Filtering channels")
        if n_channels is not None:
            for raw_set_idx, raw_set in enumerate(raw_datasets):
                n_set_channels = raw_set[0].to_numpy().shape[1]
                n_effective_channels = min((n_set_channels, n_channels))
                indices = [*range(n_effective_channels)]
                filter = RawSignalsFilterChannelIdx(indices)
                raw_datasets[raw_set_idx] = filter.fit_transform(raw_set)

        logging.debug("Filtering channels done")

        if selected_classes is not None:
            logging.debug("Selecting classes: {}".format(selected_classes))
            for raw_set_idx, raw_set in enumerate(raw_datasets):
                labels = raw_set.get_labels()
                mask = np.isin(labels, selected_classes)
                if np.all(~mask):
                    logging.error(
                        f"No selected classes {selected_classes} in dataset {experiment_name}, set {raw_set_idx}"
                    )
                    raise ValueError(
                        f"No selected classes {selected_classes} in dataset {experiment_name}, set {raw_set_idx}"
                    )
                raw_datasets[raw_set_idx] = raw_set[mask]

            logging.debug("Selecting classes done")

        extractor = wavelet_extractor2_SWT(wavelet_level=4)

        groups = []
        extr_datasets_y = []
        raw_datasets_concatenated = None

        for raw_set_idx, raw_set in enumerate(raw_datasets):
            X, y, z = extractor.fit_transform(raw_set)
            extr_datasets_y.append(y)
            groups += [raw_set_idx] * len(y)
            if raw_datasets_concatenated is None:
                raw_datasets_concatenated = deepcopy(raw_set)
            else:
                raw_datasets_concatenated += deepcopy(raw_set)

        all_y = np.concatenate(extr_datasets_y)
        skf = LeaveOneGroupOut()

        def compute(fold_idx, train_idx, test_idx):

            fold_res = []

            raw_train = raw_datasets_concatenated[train_idx]
            raw_test = raw_datasets_concatenated[test_idx]

            for filter_name, filter in tqdm(
                generate_filters().items(), desc="Filters", leave=False
            ):
                raw_train_f = filter.fit_transform(raw_train)
                raw_test_f = filter.transform(raw_test)

                X_train, y_train, _ = extractor.fit_transform(raw_train_f)
                X_test, y_test, _ = extractor.transform(raw_test_f)

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
                        (method_name, filter_name, fold_idx, y_test, y_pred)
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
                skf.split(all_y, all_y, groups=groups)
            )
        )

        records = []
        for result_sublist in results_list:
            for (
                method_name,
                filter_name,
                fold_idx,
                y_test,
                y_pred,
            ) in result_sublist:
                for y_idx, _ in enumerate(y_test):
                    records.append(
                        {
                            Dims.METHODS.value: method_name,
                            "Filter": filter_name,
                            Dims.FOLDS.value: fold_idx,
                            "y_test": y_test[y_idx],
                            "y_pred": y_pred[y_idx],
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

        cm_file_path = os.path.join(output_directory, f"{result_file_basename}_all.md")
        all_res_pdf_file_path = os.path.join(
            output_directory, f"{result_file_basename}_all.pdf"
        )
        method_spec_pdf_file_path = os.path.join(
            output_directory, f"{result_file_basename}_method.pdf"
        )

        with open(cm_file_path, "w") as all_res_file_handler:
            print("# Method specific analysis\n", file=all_res_file_handler)

            overall_df = pd.DataFrame()
            overall_cm = list()
            with PdfPages(method_spec_pdf_file_path) as method_pdf:
                for filter_name, g_f in results_df.groupby("Filter"):
                    print(f"## Filter: {filter_name}\n", file=all_res_file_handler)
                    for method_name, g in g_f.groupby(Dims.METHODS.value):
                        print(f"### {method_name}\n", file=all_res_file_handler)
                        print(f"#### Confusion matrix\n", file=all_res_file_handler)
                        cm = confusion_matrix(g["y_test"], g["y_pred"])
                        overall_cm.append(cm)
                        u_labels = np.unique(np.hstack((g["y_test"], g["y_pred"])))
                        cm_df = pd.DataFrame(cm, index=u_labels, columns=u_labels)
                        cm_df.to_markdown(all_res_file_handler)
                        print("\n", file=all_res_file_handler)

                        print(f"### classification report\n", file=all_res_file_handler)
                        cr_dict = classification_report(
                            g["y_test"], g["y_pred"], output_dict=True
                        )
                        cr_df = pd.DataFrame(cr_dict).transpose()
                        class_rows = cr_df.iloc[
                            :-3
                        ]  # assumes last 3 rows are avg metrics

                        class_specificity = specificity_score(
                            g["y_test"], g["y_pred"], average=None
                        )
                        class_sensitivity = sensitivity_score(
                            g["y_test"], g["y_pred"], average=None
                        )

                        class_gmean = geometric_mean_score(
                            g["y_test"], g["y_pred"], average=None
                        )
                        class_rows["sensitivity"] = class_sensitivity
                        class_rows["specificity"] = class_specificity
                        class_rows["g-mean"] = class_gmean
                        class_rows["f1-g"] = np.sqrt(
                            class_gmean * class_rows["f1-score"]
                        )
                        f1_norm = normalised_f1_score(g["y_test"], g["y_pred"])
                        class_rows["f1-norm"] = f1_norm
                        class_kappa = class_specific_kappa(g["y_test"], g["y_pred"])
                        class_rows["kappa"] = class_kappa

                        class_rows = class_rows[
                            [
                                "precision",
                                "recall",
                                "sensitivity",
                                "specificity",
                                "f1-score",
                                "g-mean",
                                "f1-g",
                                "f1-norm",
                                "support",
                                "kappa",
                            ]
                        ]

                        overall_df[f"{method_name}-f1"] = class_rows["kappa"]

                        crit_name = "f1-score"
                        class_rows_sorted = class_rows.sort_values(
                            crit_name, ascending=False
                        )
                        class_rows_sorted.to_markdown(all_res_file_handler)
                        print("\n", file=all_res_file_handler)
                        # TODO visualization of f1 ang g-mean
                        plt.plot(class_rows_sorted[crit_name], label=crit_name)
                        plt.plot(
                            class_rows_sorted["precision"], label="precision", alpha=0.3
                        )
                        plt.plot(class_rows_sorted["recall"], label="recall", alpha=0.3)
                        cumsums = np.cumsum(class_rows_sorted[crit_name])
                        cumcounts = np.arange(1, len(cumsums) + 1)
                        cum_mean = cumsums / cumcounts
                        plt.plot(
                            cum_mean, label=f"{crit_name}: cumulative mean", alpha=0.5
                        )
                        plt.grid(
                            True,
                            color="grey",
                            linestyle="--",
                            linewidth=0.7,
                            axis="both",
                        )
                        plt.legend()
                        plt.title(f"{filter_name},{method_name}, {crit_name}")
                        plt.xlabel("Label")
                        plt.ylabel("Criterion value")
                        method_pdf.savefig()
                        plt.close()

                        crit_name = "g-mean"
                        class_rows_sorted = class_rows.sort_values(
                            crit_name, ascending=False
                        )
                        plt.plot(class_rows_sorted[crit_name], label=crit_name)
                        plt.plot(
                            class_rows_sorted["sensitivity"],
                            label="sensitivity",
                            alpha=0.3,
                        )
                        plt.plot(
                            class_rows_sorted["specificity"],
                            label="specificity",
                            alpha=0.3,
                        )
                        cumsums = np.cumsum(class_rows_sorted[crit_name])
                        cumcounts = np.arange(1, len(cumsums) + 1)
                        cum_mean = cumsums / cumcounts
                        plt.plot(
                            cum_mean, label=f"{crit_name}: cumulative mean", alpha=0.5
                        )
                        plt.grid(
                            True,
                            color="grey",
                            linestyle="--",
                            linewidth=0.7,
                            axis="both",
                        )
                        plt.legend()
                        plt.title(f"{filter_name},{method_name}, {crit_name}")
                        plt.xlabel("Label")
                        plt.ylabel("Criterion value")
                        method_pdf.savefig()
                        plt.close()

                        # crit_name = "f1-g"
                        # class_rows_sorted = class_rows.sort_values(
                        #     crit_name, ascending=False
                        # )
                        # plt.plot(class_rows_sorted[crit_name], label=crit_name)
                        # cumsums = np.cumsum(class_rows_sorted[crit_name])
                        # cumcounts = np.arange(1, len(cumsums) + 1)
                        # cum_mean = cumsums / cumcounts
                        # plt.plot(cum_mean, label=f"{crit_name}: cumulative mean", alpha=0.5)
                        # plt.grid(
                        #     True, color="grey", linestyle="--", linewidth=0.7, axis="both"
                        # )
                        # plt.legend()
                        # plt.title(f"{filter_name},{method_name}, {crit_name}")
                        # plt.xlabel("Label")
                        # plt.ylabel("Criterion value")
                        # method_pdf.savefig()
                        # plt.close()

                        # crit_name = "f1-norm"
                        # class_rows_sorted = class_rows.sort_values(
                        #     crit_name, ascending=False
                        # )
                        # plt.plot(class_rows_sorted[crit_name], label=crit_name)
                        # cumsums = np.cumsum(class_rows_sorted[crit_name])
                        # cumcounts = np.arange(1, len(cumsums) + 1)
                        # cum_mean = cumsums / cumcounts
                        # plt.plot(cum_mean, label=f"{crit_name}: cumulative mean", alpha=0.5)
                        # plt.grid(
                        #     True, color="grey", linestyle="--", linewidth=0.7, axis="both"
                        # )
                        # plt.legend()
                        # plt.title(f"{filter_name},{method_name}, {crit_name}")
                        # plt.xlabel("Label")
                        # plt.ylabel("Criterion value")
                        # method_pdf.savefig()
                        # plt.close()

                        crit_name = "kappa"
                        class_rows_sorted = class_rows.sort_values(
                            crit_name, ascending=False
                        )
                        plt.plot(class_rows_sorted[crit_name], label=crit_name)
                        cumsums = np.cumsum(class_rows_sorted[crit_name])
                        cumcounts = np.arange(1, len(cumsums) + 1)
                        cum_mean = cumsums / cumcounts
                        plt.plot(
                            cum_mean, label=f"{crit_name}: cumulative mean", alpha=0.5
                        )
                        plt.grid(
                            True,
                            color="grey",
                            linestyle="--",
                            linewidth=0.7,
                            axis="both",
                        )
                        plt.legend()
                        plt.title(f"{filter_name},{method_name}, {crit_name}")
                        plt.xlabel("Label")
                        plt.ylabel("Criterion value")
                        method_pdf.savefig()
                        plt.close()

            with PdfPages(all_res_pdf_file_path) as pdf:
                print("# Overall CM", file=all_res_file_handler)
                overall_cm = np.asanyarray(overall_cm)
                overall_cm_sum = np.sum(overall_cm, axis=0)
                overall_cm_sum_df = pd.DataFrame(
                    overall_cm_sum, index=u_labels, columns=u_labels
                )
                order = overall_cm_sum_df.values.diagonal().argsort()[
                    ::-1
                ]  # sort descending
                overall_cm_sum_df_order = overall_cm_sum_df.iloc[order, :].iloc[
                    :, order
                ]
                overall_cm_sum_df_order.to_markdown(all_res_file_handler)
                print("\n", file=all_res_file_handler)

                labels_sorted = [u_labels[i] for i in order]

                sns.heatmap(
                    overall_cm_sum_df_order,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=True,
                )
                plt.title("Confusion Matrix (sorted by diagonal)")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                # Save as PDF
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                diagonal_vals = np.diag(overall_cm_sum_df_order.values)
                plt.plot(range(1, len(diagonal_vals) + 1), diagonal_vals, marker="o")
                plt.xticks(
                    range(1, len(diagonal_vals) + 1),
                    labels=overall_cm_sum_df_order.index,
                )
                plt.grid(True, color="grey", linestyle="--", linewidth=0.7, axis="both")
                plt.title("Scree Plot of Diagonal (Correct Counts)")
                plt.xlabel("Class")
                plt.ylabel("Correct Predictions")
                pdf.savefig()
                plt.close()

                print("# Mean F1", file=all_res_file_handler)
                mean_f1 = np.mean(overall_df, axis=1)
                mean_f1_df = pd.DataFrame(
                    mean_f1, columns=["mean-f1"], index=overall_df.index
                )
                mean_f1_df_sorted = mean_f1_df.sort_values("mean-f1", ascending=False)
                mean_f1_df_sorted.to_markdown(all_res_file_handler)

                cumulative_sum_f1_sorted = np.cumsum(mean_f1_df_sorted["mean-f1"])
                cum_counts = np.arange(1, len(u_labels) + 1)
                cumulative_mean_f1_sorted = cumulative_sum_f1_sorted / cum_counts

                plt.plot(
                    range(1, len(diagonal_vals) + 1),
                    mean_f1_df_sorted["mean-f1"],
                    marker="o",
                    label="Mean Kappa",
                )
                plt.plot(
                    range(1, len(diagonal_vals) + 1),
                    cumulative_mean_f1_sorted,
                    marker="o",
                    label="Cumulative mean Kappa",
                )
                plt.legend()
                plt.xticks(
                    range(1, len(diagonal_vals) + 1),
                    labels=overall_cm_sum_df_order.index,
                )
                plt.grid(True, color="grey", linestyle="--", linewidth=0.7, axis="both")
                plt.title("Scree Plot Kappa")
                plt.xlabel("Class")
                plt.ylabel("Kappa")
                pdf.savefig()
                plt.close()

                print("\n", file=all_res_file_handler)

                print("# Overall ranks", file=all_res_file_handler)
                ranked_df = rankdata(overall_df, axis=0, method="average")
                av_ranks_df = pd.DataFrame(
                    ranked_df.mean(axis=1), index=overall_df.index, columns=["avg-rank"]
                )
                av_ranks_sorted = av_ranks_df.sort_values("avg-rank", ascending=False)
                av_ranks_sorted.to_markdown(all_res_file_handler)
                print("\n", file=all_res_file_handler)


def normalised_f1_score(
    y_true, y_pred, labels=None, average=None, zero_division=None, N=1000
):
    f1s_randomised = []
    f1s = f1_score(y_true, y_pred, average=None)
    labels, counts = np.unique(y_true, return_counts=True)
    freqs = counts / counts.sum()
    for _ in range(N):
        y_random = np.random.choice(labels, size=len(y_true), replace=True, p=freqs)
        tmp_f1 = f1_score(y_true, y_random, average=None)
        f1s_randomised.append(tmp_f1)

    f1s_randomised = np.array(f1s_randomised)
    f1s_randomised = f1s_randomised.mean(axis=0)
    f1_normalized = (f1s - f1s_randomised) / (1 - f1s_randomised)
    return f1_normalized


def class_specific_kappa(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = cm.shape[0]
    kappas = []

    for i in range(n_classes):
        # One-vs-rest confusion matrix for class i
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        N = TP + FP + FN + TN

        po = (TP + TN) / N
        pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (N * N)

        kappa_i = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
        kappas.append(kappa_i)

    return kappas


def main():
    np.random.seed(0)
    random.seed(0)

    subjects = list([*range(1, 41)])  # ATTENTION
    experiments = list([*range(1, 4)])
    selected_classes_list = [  # ATTENTION change if needed
        ["9", "13", "16", "6", "1", "14", "7"],
        ["39", "36", "34", "40"],
        ["48", "46"],
    ]
    labels = ["stimulus", "restimulus"]

    db_name = "db3"
    db_archive_path = os.path.join(settings.DATAPATH, f"{db_name}.zip")

    data_sets = []
    for experiment, selected_classes in zip(experiments, selected_classes_list):
        for label in labels:
            data_sets.append(
                (
                    f"exp_{experiment}_{label}",
                    db_archive_path,
                    selected_classes,
                    [f".*/S{su}_E{experiment}_A1_{label}/.*" for su in subjects],
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
    Temporal features added.
    SWT used as extractor. Wavelet level 4. db6 wavelet.
    Spectral features
    """
    run_experiment(
        data_sets,
        output_directory,
        random_state=0,
        n_jobs=-1,
        overwrite=True,
        n_channels=12,  # 12 channels total, 8 around the forearm
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
