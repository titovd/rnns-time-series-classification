import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from tqdm.auto import tqdm

COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DTYPES = {
    'user': str,
    'activity': str,
    'timestamp': str,
    'x-axis': float,
    'y-axis': float,
    'z-axis': str
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--time_step", type=int, default=100)
    parser.add_argument("--seg_time_size", type=int, default=180)
    arguments = parser.parse_args()
    return arguments


def main(input_filepath, output_filepath, time_step, segment_time_size):
    data = pd.read_csv(input_filepath, header=None, names=COLUMN_NAMES, dtype=DTYPES)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data['z-axis'] = pd.to_numeric(data['z-axis'], errors='coerce')
    data.dropna(inplace=True)

    data_processed = []
    labels = []

    for i in tqdm(range(0, len(data) - segment_time_size, time_step)):
        x = data['x-axis'].values[i: i + segment_time_size]
        y = data['y-axis'].values[i: i + segment_time_size]
        z = data['z-axis'].values[i: i + segment_time_size]
        data_processed.append([x, y, z])

        label = stats.mode(data['activity'][i: i + segment_time_size])[0][0]
        labels.append(label)

    data_processed = np.asarray(data_processed, dtype=np.float32).transpose(0, 2, 1)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

    print("Processed data shape: ", data_processed.shape)
    print("Labels shape:", labels.shape)

    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    np.save(output_filepath + 'wisdm_data.npy', data_processed)
    np.save(output_filepath + 'wisdm_label.npy', labels)
    print(f"Saved wisdm_data.npy, wisdm_label.npy to {output_filepath}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args.input_path, args.output_path, args.time_step, args.seg_time_size)
