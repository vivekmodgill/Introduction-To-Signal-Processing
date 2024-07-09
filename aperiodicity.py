import numpy as np

def remove_aperiodicity(freqs, psd):
    """
    Process Power Spectral Density (PSD) data to separate the 1/f noise component
    and calculate the residuals.

    Parameters:
    freqs (array-like): Frequency values.
    psd (array-like): PSD values for each channel.

    Returns:
    tuple: A tuple containing:
        - aperiodic_comp (np.ndarray): The fitted 1/f noise models for each channel.
        - periodic_psd (np.ndarray): The residuals after subtracting the 1/f noise.
    """
    # Initialize lists to store results
    aperiodic_models = []
    residuals        = []

    # Ensure no invalid values in freqs
    freqs     = np.where(np.isfinite(freqs), freqs, 1e-10)
    log_freqs = np.log(freqs + 1e-10)

    for channel_idx in range(len(psd)):  # Loop over channels
        # Ensure no invalid values in psd[channel_idx]
        psd_channel = np.where(np.isfinite(psd[channel_idx]), psd[channel_idx], 1e-10)
        log_psd     = np.log(psd_channel + 1e-10)

        try:
            # Fit a linear model to the log-log PSD data
            fit_params = np.polyfit(log_freqs, log_psd, deg=1)

            # Calculate the fitted model and residuals
            fitted_model = np.exp(np.polyval(fit_params, log_freqs))
            residual     = psd_channel - fitted_model

            # Store the fitted model and residuals
            aperiodic_models.append(fitted_model)
            residuals.append(residual)
        except np.linalg.LinAlgError as err:
            print(f"Linear algebra error for channel {channel_idx}: {err}")
            aperiodic_models.append(np.full_like(psd_channel, np.nan))
            residuals.append(np.full_like(psd_channel, np.nan))

    # Convert lists to numpy arrays
    
    aperiodic_comp = np.array(aperiodic_models)
    periodic_psd   = np.array(residuals)

    return aperiodic_comp, periodic_psd
