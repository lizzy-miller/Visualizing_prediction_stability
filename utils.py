import numpy as np
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
        # Normalize delta and eta so they have the same scale as theta_opt
        delta = delta * (np.linalg.norm(theta_opt) / (np.linalg.norm(delta) + 1e-8))
        eta = eta * (np.linalg.norm(theta_opt) / (np.linalg.norm(eta) + 1e-8)) 
                                                  
    return theta_opt + alpha * delta + beta * eta
