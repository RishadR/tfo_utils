# TFO Utils
A set of utility functions for all TFO-related projects including data exploration, visualization, etc.

## Spectrum Visualizer
Creates 2D Spectrograms.

### Instructions
1. Place the demodulated PPG csv file inside the data folder - it should now appear as an option
2. The PPG file should have data along its columns and different channels along the rows
3. Options - High Pass: A 5th order Butterworth no-phase high pass filter
4. Options - Low Pass : A 5th order Butterworth no-phase low pass filter
5. Options - Window Denoise: Denoise the spectrogram using per-window energy median. Any window with an energy median * threshold above of below the median is replaced by the window above it. This is meant to kill motion artifacts. Lower the threshold to increase power. (Also see "Options - Spectral Frequency limits")
6. Options - Spectral Frequency limits : The spectrograms are cropped to this limit *before*  window denoise computations
7. Options - PGA Gain : Does nothing for now
8. Options - Verbose : Prints out how many windows got replaced(denoised) and how many NaN values were replaced with top imputation
9. All options have their default values set by defaults YAML
10. The spectrograms plot in dB with the same color scale across all plots
11. Only plots the first 10 channels in a 2 x 5 grid
