from utils import *
from ml_utils import XrayTF
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')

    parser.add_argument("-g",   "--Gender",       type=str,
                        help="Which gender should you train", default="M")
    parser.add_argument("-xp",  "--XrayPosition", type=str,
                        help="Available Options are PA and AP", default="PA")
    parser.add_argument("-img", "--ImageSize",    type=int,
                        help="Image size to be trained", required=True)
    parser.add_argument("-lq",  "--LabelQuota",   type=int,
                        help="For the label targets, what is the count cutoff?", default=10000)
    parser.add_argument("-b",   "--BatchSize",    type=int,
                        help="Batch size for training", default=32)
    parser.add_argument("-csv", "--CsvFile",      type=str,
                        help="This should be located within ../sheet/", required=True)

    args = parser.parse_args()

    GENDER = args.Gender
    POSITION = args.XrayPosition
    csv_file = args.CsvFile  # normalized_xray_data_with_no_finding.csv
    IMAGE_SIZE = args.ImageSize
    BATCH_SIZE = args.BatchSize
    LABEL_QUOTA = args.LabelQuota

    csv_file_full_path = f'/content/xray_code/sheet/{csv_file}'

    # take existing if it exist
    if os.path.exists(csv_file_full_path):
        df = pd.read_csv(csv_file_full_path)
        xray_class = XrayTF(df, IMAGE_SIZE, BATCH_SIZE)
    else:
        df = get_data_sheet()
        df = normalize_data_frame(df)
        xray_class = XrayTF(df, IMAGE_SIZE, BATCH_SIZE)
        xray_class.prepend_image_full_path()
        xray_class.df.to_csv(csv_file_full_path, index=False)

    # drop rows based on label column value counts
    xray_class.prune_based_on_quota(LABEL_QUOTA)
    # make all labels equal based on the lowest label.
    xray_class.balance_all_labels()

    # Filter dataframe based on gender and position
    xray_class.df = xray_class.df[xray_class.df["Patient Gender"] == GENDER]
    xray_class.df = xray_class.df[xray_class.df["View Position"] == POSITION]

    # Split them into training and validation using NUM_IMAGES
    train_df, valid_df = xray_class.get_test_train_split_data(
        len(xray_class.df))

    train_data = xray_class.generate_image(train_df)
    val_data = xray_class.generate_image(valid_df)

    # (number of epoc, early stopping)
    model = xray_class.train_model(5, train_data, val_data)
    xray_class.save_model(model, "all_sickness")
