import torch
from torch.utils.data import Dataset
import pandas as pd

class DeepARDataset(Dataset):
    def __init__(self, 
                train_df: pd.DataFrame,
                max_encoder_length: int,
                max_prediction_length: int, 
                static_covariates: list,
                time_varying_covariates: list,
                group_ids: list
                ):
        self.train_df = train_df
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.static_covariates = static_covariates
        self.time_varying_covariates = time_varying_covariates
        self.group_ids = group_ids
        self.series_list = []

        grouped = self.train_df.groupby(self.group_ids)
        for _, group_df in grouped:
            relevant_data = group_df[self.time_varying_covariates].values
            self.series_list.append(torch.tensor(relevant_data, dtype=torch.float32))

        self.indices = []
        total_window = self.max_encoder_length + self.max_prediction_length
        for series_idx, series_data in enumerate(self.series_list):
            seq_len = len(series_data)
            num_windows = seq_len - total_window + 1
            if num_windows > 0:
                for start_pos in range(num_windows):
                    self.indices.append((series_idx, start_pos))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        series_idx, start_pos = self.indices[index]
        series = self.series_list[series_idx]
        total_window = self.max_encoder_length + self.max_prediction_length
        data_slice = series[start_pos : start_pos + total_window]
        past = data_slice[: self.max_encoder_length]
        future = data_slice[self.max_encoder_length :]
        return past, future
