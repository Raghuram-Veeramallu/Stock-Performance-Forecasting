import os
import yaml

import pandas as pd

# create a directory if not exists
def create_folder_if_not_exists(file_path: str):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

# function to persist the data from dataframe into a csv
def export_data_frame_to_csv(dataframe: pd.DataFrame, output_fp: str, keep_indices: bool = False):
    create_folder_if_not_exists(output_fp)
    dataframe.to_csv(output_fp, index=keep_indices)

def read_data_from_pd_dataframe(filepath: str):
    return pd.read_csv(filepath)

def load_configs_from_yaml(yaml_filepath):
    if os.path.exists(yaml_filepath):
        with open(yaml_filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError('Config file not found!')

    return config
