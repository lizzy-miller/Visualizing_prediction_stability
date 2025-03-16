# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import torch 
import torch.nn as nn
import torch.optim as optim  
import random
from sklearn.model_selection import train_test_split
import sys  
import os

os.chdir('/Users/lizzy/Desktop/Visualizing_prediction_stability')
# %%
# Generate Data

# Parameters
beta_0 = -3  # Intercept
beta_1 = 0.5  # Slope for x
beta_2 = 1  # Slope for z
beta_3 = 0.1 # slope for x^2
beta_4 = -0.05 # sloe for z^2
beta_5 = -0.05 # slope for x*z
beta_6 = 0.02 # slope for x^3 
beta_7 = -0.3 # slope for w # unobserved
beta_8 = 0.2
n = 1000000  # Number of samples

# Define means and covariance
mu = [1, 0]  # Mean values for x and z
std_x = 2
std_z = 1
corr_xz = 0.6
sigma = [[std_x**2, corr_xz * std_x * std_z], 
         [corr_xz * std_x * std_z, std_z**2]]  

# Generate synthetic data
x, z = np.random.multivariate_normal(mean=mu, cov=sigma, size=n).T
w = np.random.normal(loc = 0.1, scale = 1, size = n) # unobserved variable
x_c = (x - np.mean(x)) / np.std(x)
z_c = (z - np.mean(z)) / np.std(z)
w_2 = 0.8 * x + w

# Compute xb
error = np.random.normal(loc=0, scale=0.25, size=n)
xb = (beta_0 
      + beta_1 * x_c 
      + beta_2 * z_c 
      + beta_3 * x_c**2 
      + beta_4 * z_c**2 
      + beta_5 * x_c * z_c 
      + beta_6 * x_c**3
      + error)
xb += beta_7 * w
xb += beta_8 * w_2


# Compute probabilities and sample y
p = 1 / (1 + np.exp(-xb))

noise = np.random.normal(loc = 0, scale = 0.01, size = n)
p_noisy = np.clip(p + noise, 0, 1).ravel()
y = np.random.binomial(n=1, p=p_noisy.squeeze(), size=n)

ids = np.arange(1, n + 1)  # IDs from 1 to n

data = pd.DataFrame({'id': ids, 'x': x, 'z': z, 'y': y})

data.head(10)

data_ = data


# %%
data['y'].value_counts()

# %%
# Showing Optimal Distribution

x_seq = np.linspace(data['x'].min(), data['x'].max(), 100)
z_seq = np.linspace(data['z'].min(), data['z'].max(), 100)
x_grid, z_grid = np.meshgrid(x_seq, z_seq)
XZ_flat = np.column_stack([x_grid.ravel(), z_grid.ravel()])


# Compute corrected true probabilities with quadratic terms
xb_true = (beta_0 
           + beta_1 * XZ_flat[:, 0] 
           + beta_2 * XZ_flat[:, 1]
           + beta_3 * XZ_flat[:, 0]**2
           + beta_4 * XZ_flat[:, 1]**2
           + beta_5 * XZ_flat[:, 0] * XZ_flat[:, 1]
           + beta_6 * XZ_flat[:, 0]**3) 

y_pred_true = 1 / (1 + np.exp(-xb_true)).reshape(100, 100)
# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, z_grid, y_pred_true, cmap='gray', alpha=0.7)
ax.plot_wireframe(x_grid, z_grid, y_pred_true, color='black', alpha=0.3, linewidth=0.5, linestyle="dashed")
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('P(y=1)')
ax.set_title('Optimal Probability Surface')
ax.view_init(elev=25, azim=310)
plt.show()

# %%
# Split data, Sample size 500

sample_size = 500
num = np.random.randint(0, 2**32) 
np.random.seed(num)
sampled_df, _ = train_test_split(data, train_size=sample_size, stratify=data["y"], random_state=num)# %%

# %% 
# Fit logistic model
X = sampled_df[["x", "z"]].values
y = sampled_df["y"].values
logreg = LogisticRegression(penalty=None, solver="lbfgs")
logreg.fit(X, y)
log_loss(y, logreg.predict_proba(X)[:, 1])
# %%
from utils import tau_1d, tau_2d



# extract model parameters
theta_opt = np.hstack([logreg.intercept_, logreg.coef_.ravel()])
delta = np.random.randn(*theta_opt.shape)
eta = np.random.randn(*theta_opt.shape)

alpha_values = np.linspace(-20, 20, 100)
theta_star = theta_opt + np.random.normal(scale=1.0, size=theta_opt.shape) # randomly selecting theta_star
# %%
lv_log_interpolated = []
log_interpolation = {}

for alpha in alpha_values:
    theta_alpha = tau_1d(alpha = alpha, theta = theta_opt, theta_star = theta_star, normalize = True)
    log_model_interpolated = LogisticRegression(penalty='none', solver='lbfgs')
    log_model_interpolated.intercept_ = np.array([theta_alpha[0]])
    log_model_interpolated.coef_ = np.array([theta_alpha[1:]])
    log_model_interpolated.classes_ = np.array([0, 1])
    probs = log_model_interpolated.predict_proba(X)[:, 1]
    lv_log_interpolated.append(log_loss(y, probs))
    log_interpolation[alpha] = {
        "intercept": theta_alpha[0],
        "coef": theta_alpha[1:],
        "loss": log_loss(y, probs)
    }

log_interpolation = pd.DataFrame.from_dict(log_interpolation, orient='index')

plt.figure(figsize=(6, 6))
plt.plot(alpha_values, lv_log_interpolated)
plt.xlabel(r"$\alpha$")
plt.ylabel("Log-loss")
plt.title("Interpolated 1D Log-loss")
plt.grid(True)
plt.show()
# %%
loss_values
# %%
