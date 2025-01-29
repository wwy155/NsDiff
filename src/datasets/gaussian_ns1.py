import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import torch
from typing import Any, Callable, List, Optional
import os
import resource
from torch_timeseries.core.dataset.dataset import Dataset, TimeSeriesDataset


class GaussianNS1(TimeSeriesDataset):
    """
    The collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore ranging from 1990 to 2016.
    
    The raw data is collected from https://github.com/laiguokun/multivariate-time-series-data.

    Attributes:
        name (str): Name of the dataset.
        num_features (int): Number of features in the dataset.
        freq (str): Frequency of the data points (daily).
        length (int): Length of the dataset.

    Methods:
        download():
            Downloads and extracts the dataset.
        _load():
            Loads the dataset into a NumPy array.
    """

    
    name: str = 'GaussianNS'
    num_features: int = 2
    freq: str = 'yd'  # daily data
    length: int = 7588
    
    def generate_synthetic_data(self) -> np.ndarray:
        """
        Generates synthetic data with linearly increasing means and variances.
        Means increase linearly from 1 to 10, variances increase linearly from 1 to 10.
        
        Returns:
            np.ndarray: The generated synthetic dataset.
        """        
        # Linearly increasing means and variances over time
        # means = np.linspace(1, 10, self.length)  # means from 1 to 10
        # variances = np.linspace(1, 5, self.length)  # variances from 1 to 10
        
        means = np.linspace(1, 20, self.length)  # means from 1 to 10
        stddev = np.linspace(1, 20, self.length)  # variances from 1 to 10

        
        # Generate synthetic data with different features
        data = np.zeros((self.length, self.num_features))
        for t in range(self.length):
            for i in range(self.num_features):
                # For each feature, add an offset (i + 1) to the mean
                feature_mean = means[t] + (i)  # feature-specific mean (e.g., +1, +2, ...)
                data[t, i] = np.random.normal(loc=feature_mean, scale=stddev[t], size=1)
        
        return data

    
    
    def _load(self) -> np.ndarray:
        """
        Loads the synthetic dataset into a NumPy array and adds dates.

        Returns:
            np.ndarray: The synthetic dataset with dates.
        """
        self.data = self.generate_synthetic_data()
        
        # Create a date range starting from '1990-01-01' with daily frequency
        self.dates = pd.date_range(start="1990-01-01", periods=self.length, freq='D')
        self.dates =  pd.DataFrame({'date':self.dates})
        # Create DataFrame for the data
        self.df = pd.DataFrame(self.data, columns=[f'feature_{i+1}' for i in range(self.num_features)], index=self.dates)
        
        return self.data

    def download(self) -> None:
        pass
