import datetime
import itertools
import os
import numpy as np
import pandas as pd

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

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    read_signals_from_dirs,
)

from tqdm import tqdm

import random

from emg_experiment_simple import settings
from emg_experiment_simple.tools import logger
import logging

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


def run_experiment(
    input_data_dir_list,
    output_directory,
    n_channels=None,
    progress_log_handler=None,
    comment_str="",
):

    os.makedirs(output_directory, exist_ok=True)

    comment_file = os.path.join(output_directory, "comment.txt")
    with open(comment_file, "w") as f:
        f.write(comment_str)
        f.write("Start time: {}\n".format(datetime.datetime.now()))
        f.write("\n")

        f.write(f"n_channels: {n_channels}\n")
        f.write("\n")

    extractors_dict = create_extractors()

    for in_dir in tqdm(input_data_dir_list, desc="Data sets"):

        set_name = os.path.basename(in_dir)
        
        if not os.path.exists(in_dir):
            logging.debug("Skipping {} !".format(set_name))
            continue

        pre_set = read_signals_from_dirs(in_dir)
        raw_set = pre_set["accepted"]

        if n_channels is not None:
            n_set_channels = raw_set[0].to_numpy().shape[1]
            n_effective_channels = min((n_set_channels, n_channels))
            indices = [*range(n_effective_channels)]
            filter = RawSignalsFilterChannelIdx(indices)
            raw_set = filter.fit_transform(raw_set)

        for extractor_name, extractor in tqdm(
            extractors_dict.items(), desc="Extractors: ", file=progress_log_handler
        ):
            result_file_path = os.path.join(output_directory, "{}_{}.csv".format(set_name, extractor_name))
            
            X,y,_ = extractor.fit_transform(raw_set)
            
            attrib_names = [f"A_{i}" for i in range(X.shape[1])]

            attr_df = pd.DataFrame(X,columns=attrib_names)
            label_df = pd.DataFrame(y,columns=["Label"])
            all_df = pd.concat([attr_df,label_df],axis=1)
            all_df.to_csv(result_file_path, index=False,sep=';')


def main():
    np.random.seed(0)
    random.seed(0)

    data_path0B = os.path.join(settings.DATAPATH, "MK_10_03_2022_EMG")
    data_path0C = os.path.join(settings.DATAPATH, "KrzysztofJ_all_EMG")
    data_path0D = os.path.join(settings.DATAPATH, "Andrzej_17_03_2022_EMG")
    data_path0E = os.path.join(settings.DATAPATH, "Andrzej_24_04_2023-ALL_EMG")
    data_path0F = os.path.join(settings.DATAPATH, "AW-13_03_2025")
    data_path0G = os.path.join(settings.DATAPATH, "Andrzej_18_05_2023_EMG")
    data_path0H = os.path.join(settings.DATAPATH, "AW-03_03_2025")
    data_path0I = os.path.join(settings.DATAPATH, "Barbara_13_05_2022_AB")
    


    # data_sets = [data_path0B, data_path0C, data_path0D, data_path0E, data_path0F, data_path0G, data_path0H, data_path0I]
    # data_sets = [ os.path.join( settings.DATAPATH, "tsnre_windowed","A{}_Force_Exp_low_windowed".format(i)) for i in range(1,10) ]

    subjects = list([*range(1,12)])
    experiments = list([*range(1,4)])
    labels = ['stimulus', 'restimulus']
    data_sets = [ os.path.join( settings.DATAPATH, "db3",f"S{su}_E{ex}_A1_{la}") for su,ex,la in itertools.product(subjects,experiments,labels) ]

    output_directory = os.path.join(
        settings.EXPERIMENTS_RESULTS_PATH,
        "./extracted_features/",
    )
    os.makedirs(output_directory, exist_ok=True)

    log_dir = os.path.dirname(settings.EXPERIMENTS_LOGS_PATH)
    log_file = os.path.splitext(os.path.basename(__file__))[0]
    logger(log_dir, log_file, enable_logging=False)

    progress_log_path = os.path.join(output_directory, "progress.log")
    progress_log_handler = open(progress_log_path, "w")

    comment_str = """
    Simple feature extraction.
    """
    run_experiment(
        data_sets,
        output_directory,
        n_channels=8,
        progress_log_handler=progress_log_handler,
        comment_str=comment_str,
    )


if __name__ == "__main__":
    main()
