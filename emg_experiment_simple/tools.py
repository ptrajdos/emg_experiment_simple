from datetime import datetime
import logging
import os

def logger(logging_dir_path,log_file_name="logfile",enable_logging:bool=True):

    if enable_logging:
        date_string = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

        log_format_str = "%(asctime)s;%(levelname)s;%(message)s"
        log_date_format = "%Y-%m-%d %H:%M:%S"
        log_filename = "{}_{}.log".format(log_file_name,date_string)
        log_file_path =  os.path.join(logging_dir_path, log_filename)
        logging.basicConfig(filename=log_filename, level=logging.DEBUG, format=log_format_str, datefmt=log_date_format)
        logging.captureWarnings(True)

