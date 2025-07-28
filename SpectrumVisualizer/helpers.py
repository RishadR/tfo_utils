"""
Functions for the Spectrum Visualizer application.
"""

from typing import Literal
import numpy as np
from scipy.signal.windows import hann  # This matches the MATLAB 'hanning/periodic' window
from scipy.signal import ShortTimeFFT


# Spectrogram Plotter - Remake of K. Vali, Aprl 18, 2023
# Uses scipy's STFT function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html
def spectrogram_plot_2024_func(
    data: np.ndarray, fs: float = 80.0, overlap_percentage: float = 0.75, window_length_seconds: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a spectrogram plot using ShortTimeFFT with a Hann window and linear detrending.

    Args:
        data (np.ndarray): Input signal data (1D)
        fs (float): Sampling frequency in Hz.
        overlap_percentage (float): Percentage of overlap between consecutive windows.
        window_length_seconds (float): Length of the window in seconds.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the spectrogram data and frequency bins.
            - np.ndarray: The spectrogram data with time along the vertical axis and frequencies along the horizontal axis.
            - np.ndarray: The frequency bins corresponding to the spectrogram data.
    """
    noiseBwHz = 6  # For future use
    spectrumtype: Literal["magnitude", "psd"] = "magnitude"
    window_length_samples = int(window_length_seconds * fs)
    # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html
    win = hann(window_length_samples, sym=False)  # Create a periodic Hann window
    nover = int(window_length_samples * overlap_percentage)

    STFTGenerator = ShortTimeFFT(win, window_length_samples - nover, fs, scale_to=spectrumtype)

    return STFTGenerator.stft_detrend(data, "linear").T, STFTGenerator.f

def window_median_denoise(stft_data: np.ndarray, energy_threshold: float = 0.6) -> tuple[np.ndarray, list[int]]:
    """
    Denoise motion artifacts from STFT data using a median energy threshold.
    
    Algorithm:
        1. Calculate the energy of each window (Each row in `stft_data`).
        2. Determine the median energy
        3. Mask out windows with energy above/below the threshold
        4. Impute masked windows with the a non-masked window above it
    Args:
        stft_data (np.ndarray): STFT data with shape (n_windows, n_frequencies).
        energy_threshold (float): Threshold for energy masking.
    
    Returns:
        A tuple containing:
        - np.ndarray: Denoised STFT data.
        - list[int]: Indices of windows that were masked out.
    """
    # Calculate the energy of each window
    energy = np.sum(np.abs(stft_data) ** 2, axis=1)
    
    # Calculate the median energy
    median_energy = np.median(energy)

    # Create a mask for windows with energy above/below the threshold
    mask = np.abs(energy - median_energy) >= energy_threshold * median_energy

    # Get indices of masked windows
    masked_indices = np.where(mask)[0].tolist()
    
    # Impute masked windows with the last non-masked window above it
    if 0 in masked_indices:
        # Set the first masked window to the median window (by energy)
        median_idx = int(np.argsort(np.abs(energy - median_energy))[0])
        stft_data[0] = stft_data[median_idx]
    
    for idx in masked_indices:
        if idx > 0:
            stft_data[idx] = stft_data[idx - 1]
    
    return stft_data, masked_indices

def impute_with_left(time_series: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Impute NaN values in a time series with the last valid value to the left.
    
    Args:
        time_series (np.ndarray): Input time series with NaN values.
    
    Returns:
        a tuple containing:
        - np.ndarray: Time series with NaN values imputed.
        - list[int]: Indices of the NaN values that were imputed.
    """
    # Create a copy of the time series to avoid modifying the original
    imputed_series = np.copy(time_series)
    
    # Find indices where the series is NaN
    nan_indices = np.where(np.isnan(imputed_series))[0].tolist()
    
    # Early return if there are no NaNs    
    if len(nan_indices) == 0:
        return imputed_series, []

    # Impute NaNs with the last valid value to the left
    for idx in nan_indices:
        if idx > 0:
            imputed_series[idx] = imputed_series[idx - 1]

    return imputed_series, nan_indices
