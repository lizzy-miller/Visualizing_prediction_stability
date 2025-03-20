import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming

def tau_1d(alpha, theta, theta_star, normalize = False):
    """
    Performs 1D linear interpolation between initial parameters (`theta`) and 
    perturbed parameters (`theta_star`), with optional filter-wise normalization.
    
    """
    if normalize:
        theta_star_norm = np.copy(theta_star) 
        # Normalize the difference between theta_star and theta
        norm_factor = np.linalg.norm(theta_star_norm - theta) + 1e-8  # Avoid division by zero
        theta_star_norm = theta + ((theta_star_norm - theta) / norm_factor) * np.linalg.norm(theta)

        return (1 - alpha) * theta + alpha * theta_star_norm
    else:
        return (1 - alpha) * theta + alpha * theta_star

def tau_2d(alpha, beta, theta_opt, delta, eta, normalize = False):
    """
    Performs 2D linear interpolation along two independent directions
    with optional filter-wise normalization.

    """
    if normalize:
        # Normalize delta and eta to unit vectors before scaling
        delta = delta / (np.linalg.norm(delta) + 1e-8)  # Avoid division by zero
        eta = eta / (np.linalg.norm(eta) + 1e-8)        # Avoid division by zero
        delta = delta * np.linalg.norm(theta_opt)
        eta = eta * np.linalg.norm(theta_opt)
                                                  
    return theta_opt + alpha * delta + beta * eta


def bin_rows(df, coef_threshold=0.01, index_threshold=0.01):
    """
    Bins rows where:
    - Binary predictions match exactly (Hamming distance is 0).

    Args:
        df (pd.DataFrame): DataFrame with "binary_predictions" column containing lists/arrays.
        coef_threshold (float): Maximum allowed difference between coefficient values.
        index_threshold (float): Maximum allowed difference between index values.

    Returns:
        pd.DataFrame: DataFrame with an added "bin_group" column.
    """
    bin_groups = {}
    predictions_saved = []
    
    for current_row in df.itertuples():
        binary_predictions_current = current_row.binary_predictions.ravel()  
        found_bin = False
        for saved_index, binary_predictions_saved in enumerate(predictions_saved):
            binary_predictions_saved = binary_predictions_saved.ravel() 
            if hamming(binary_predictions_current, binary_predictions_saved) == 0:
                bin_groups[current_row.Index] = saved_index  
                found_bin = True
                break
        if not found_bin:
            bin_groups[current_row.Index] = len(predictions_saved)  # Create new bin
            predictions_saved.append(binary_predictions_current)
    
    df["bin"] = df.index.map(bin_groups)
    return df


def hamming_distance_bins(df, binary_col='binary_predictions', bin_col = "bin", reference_row = None):
    """
    Calculates the Hamming distance between rows in the same bin group.
    df needs to have 'loss' column and 'bin' column.
    """
    if reference_row is None:
        reference_row = df["loss"].idxmin() # Automatically find row with lowest loss
    reference_binary_predictions = df.loc[reference_row, binary_col]
    
    hamming_distances = []
    
    for bin_group, group_data in df.groupby(bin_col):
        group_hamming_distances = []
        for _, row in group_data.iterrows():
            current_binary_predictions = row[binary_col]
            hamming_distance = np.sum(reference_binary_predictions != np.array(current_binary_predictions))
            group_hamming_distances.append(hamming_distance)
        hamming_distances.extend(group_hamming_distances)
    
    df["hamming_distance"] = hamming_distances
    return df