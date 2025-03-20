# %% 
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.spatial.distance import hamming
import torch
import torch.nn as nn
import torch.optim as optim
from utils import tau_1d, tau_2d, bin_rows, hamming_distance_bins

os.chdir('/Users/lizzy/Desktop/Visualizing_prediction_stability')

# Load Data
data = pd.read_csv('data/log_synthetic_data.csv')
data_ = data
data.head(10)

# %% 
# Split data into training and test sets
data, test_data = train_test_split(data, test_size=1000, stratify=data['y'], random_state=0)

# Optimal Simple Logistic Regression
X = data[['x', 'z']].values
y = data['y'].values
optimal_model_log = LogisticRegression(penalty=None, solver='lbfgs')
optimal_model_log.fit(X, y)
optimal_loss_log_training = log_loss(y, optimal_model_log.predict_proba(X)[:, 1])
optimal_loss_log_test = log_loss(test_data['y'], optimal_model_log.predict_proba(test_data[['x', 'z']].values))
num_params_simple = optimal_model_log.coef_.size + optimal_model_log.intercept_.size
print(f"Number of Training Data: {len(data)}")
print(f"Number of Test Data: {len(test_data)}")
print(f"Number of parameters in simple/optimal logistic regression model: {num_params_simple}")
print(f"Optimal Log-loss Training: {optimal_loss_log_training:.4f}")
print(f"Optimal Log-loss Test: {optimal_loss_log_test:.4f}")

# %% 
# Split data, Sample size 500
sample_size = 500
num = np.random.randint(0, 2**32) 
np.random.seed(num)
sampled_df, _ = train_test_split(data, train_size=sample_size, stratify=data["y"], random_state=num)

# Fit Simple Logistic Model
X = sampled_df[["x", "z"]].values
y = sampled_df["y"].values
X_test = test_data[["x", "z"]].values
y_test = test_data["y"].values
logreg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000000)
logreg.fit(X, y)
loss_log_training = log_loss(y, logreg.predict_proba(X)[:, 1])
loss_log_testing = log_loss(y_test, logreg.predict_proba(X_test)[:, 1])
accuracy_training = (y == logreg.predict(X)).mean()
accuracy_testing = (y_test == logreg.predict(X_test)).mean()
print(f"Number of Training Data: {len(sampled_df)}")
print(f"Number of Test Data: {len(test_data)}")
print(f"Number of parameters in simple logistic regression model: {num_params_simple}")
print(f"Training log-loss: {loss_log_training:.4f}")
print(f"Testing log-loss: {loss_log_testing:.4f}")
print(f"Training accuracy: {accuracy_training:.4f}")
print(f"Testing accuracy: {accuracy_testing:.4f}")

# %% 
# Fit Overparameterized Logistic Model
# This doesn't work, may need to run in Rivanna... 
# %% 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data
poly = PolynomialFeatures(degree=500, interaction_only=False)
logreg_over = make_pipeline(poly, LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000000000))
logreg_over.fit(X_scaled, y)
loss_log_over_training = log_loss(y, logreg_over.predict_proba(X_scaled)[:, 1])
loss_log_over_test = log_loss(y_test, logreg_over.predict_proba(X_test_scaled)[:, 1])
num_params_over = (
    logreg_over.named_steps['logisticregression'].coef_.size +
    logreg_over.named_steps['logisticregression'].intercept_.size
)
print(f"Number of parameters in overparameterized logistic regression model: {num_params_over}")
print(f"Overparameterized Log-loss Training: {loss_log_over_training:.4f}")
print(f"Overparameterized Log-loss Test: {loss_log_over_test:.4f}")

# %% 
# Interpolation
theta_opt = np.hstack([logreg.intercept_, logreg.coef_.ravel()])
delta = np.random.randn(*theta_opt.shape)
eta = np.random.randn(*theta_opt.shape)
alpha_values = np.linspace(-10, 10, 1000)
if 0 not in alpha_values:
    alpha_values = np.unique(np.append(alpha_values, 0)) # This ensures alpha = 0 included
theta_star = theta_opt + np.random.normal(scale=2.0, size=theta_opt.shape) # randomly selecting theta_star
lv_log_interpolated = []
lv_log_interpolated_test = []
ac_log_interpolated = []
ac_log_interpolated_test = []
log_interpolation = {}
log_interpolation_test = {}
for alpha in alpha_values:
    theta_alpha = tau_1d(alpha=alpha, theta=theta_opt, theta_star=theta_star, normalize=True)
    log_model_interpolated = LogisticRegression(penalty='none', solver='lbfgs')
    log_model_interpolated.intercept_ = np.array([theta_alpha[0]])
    log_model_interpolated.coef_ = np.array([theta_alpha[1:]])
    log_model_interpolated.classes_ = np.array([0, 1])
    probs = log_model_interpolated.predict_proba(X)[:, 1]
    probs_test = log_model_interpolated.predict_proba(X_test)[:, 1]
    lv_log_interpolated.append(log_loss(y, probs))
    lv_log_interpolated_test.append(log_loss(y_test, probs_test))
    ac_log_interpolated.append((y == (probs >= 0.5).astype(int)).mean())  # Calculate accuracy
    ac_log_interpolated_test.append((y_test == (probs_test >= 0.5).astype(int)).mean())  # Calculate accuracy
    log_interpolation[alpha] = {
        "inter": theta_alpha[0],
        "coefficients": theta_alpha[1:],
        "loss": log_loss(y, probs),
        "risk_predictions": probs,
        "binary_predictions": (probs >= 0.5).astype(int) # threshold = 0.5
    }
    log_interpolation_test[alpha] = {
        "inter": theta_alpha[0],
        "coefficients": theta_alpha[1:],
        "loss": log_loss(y_test, probs_test),
        "risk_predictions": probs_test,
        "binary_predictions": (probs_test >= 0.5).astype(int) # threshold = 0.5
    }
log_interpolation = pd.DataFrame.from_dict(log_interpolation, orient='index')
log_interpolation_test = pd.DataFrame.from_dict(log_interpolation_test, orient='index')

# Create a dataframe to store the estimated risks
# Collect binary predictions in a dictionary
binary_preds_train_dict = {f'risk_alpha_{alpha}': log_interpolation.loc[alpha, 'binary_predictions'] for alpha in alpha_values}
binary_preds_test_dict = {f'test_risk_alpha_{alpha}': log_interpolation_test.loc[alpha, 'binary_predictions'] for alpha in alpha_values}
binary_preds_train = pd.DataFrame(binary_preds_train_dict, index=sampled_df['id'])
binary_preds_test = pd.DataFrame(binary_preds_test_dict, index=test_data['id'])

# %% 
# Loss and Accuracy 

# First y-axis for loss
plt.figure(figsize=(10, 10))
plt.plot(alpha_values, lv_log_interpolated, label="Training Log-loss", color = "#FF0000")
plt.plot(alpha_values, lv_log_interpolated_test, color="#E50000", linestyle='--', label="Testing Log-loss")
plt.axhline(y=optimal_loss_log_training, color='black', linestyle='--', label=f'Optimal Log-loss Training: {optimal_loss_log_training:.2f}')
#plt.axhline(y=optimal_loss_log_test, color='grey', linestyle='--', label=f'Optimal Log-loss Test: {optimal_loss_log_test:.4f}')
plt.scatter(0, loss_log_training, label=f'Fitted Log-loss Training: {loss_log_training:.2f}', color='#FF0000')
#plt.scatter(0, loss_log_testing, label=f'Fitted Log-loss Test: {loss_log_testing:.2f}', color='#E50000')
plt.xlabel(r"$\alpha$")
plt.ylabel("Log-loss")

# second y-axis for accuracy
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(alpha_values, ac_log_interpolated, label="Training Accuracy", color='#15B01A', alpha=0.8)
ax2.plot(alpha_values, ac_log_interpolated_test, color="#008000", linestyle='--', label="Testing Accuracy", alpha=0.8)
ax2.scatter(0, accuracy_training, label=f'Fitted Training Accuracy: {accuracy_training:.2f}', color='#15B01A')
ax2.set_ylabel("Accuracy")

plt.title("Logistic Regression Loss and Accuracy\nAcross 1D Interpolation of Model Parameters")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
#plt.grid(True)
plt.savefig("logistic_regression_loss_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# %% 
# Adding deviation from fitted model
tolerance = 0.01
filtered_rows = log_interpolation[(log_interpolation["loss"] >= loss_log_training - tolerance) & 
                                  (log_interpolation["loss"] <= loss_log_training + tolerance)]

print(f'There are {len(filtered_rows)} alpha values within the tolerance of {tolerance} from the fitted model.')
# Fixing the calculation of min and max filtered alpha
min_filtered_alpha = filtered_rows["loss"].idxmin() if not filtered_rows.empty else None
max_filtered_alpha = filtered_rows["loss"].idxmax() if not filtered_rows.empty else None

# Loss and Accuracy Plot
plt.figure(figsize=(10, 10))
plt.plot(alpha_values, lv_log_interpolated, label="Training Log-loss", color="#FF0000")
plt.plot(alpha_values, lv_log_interpolated_test, color="#E50000", linestyle='--', label="Testing Log-loss")

# Add vertical lines only if min and max filtered alpha are not None
if min_filtered_alpha is not None:
    plt.axvline(x=min_filtered_alpha, color='grey', linestyle='--', label=f'Lower Bound (Loss ± {tolerance})', alpha=0.5)
if max_filtered_alpha is not None:
    plt.axvline(x=max_filtered_alpha, color='grey', linestyle='--', label=f'Upper Bound (Loss ± {tolerance})', alpha=0.5)

# Scatter points for fitted losses
plt.scatter(0, loss_log_training, label=f'Fitted Log-loss Training: {loss_log_training:.2f}', color='#FF0000')

plt.xlabel(r"$\alpha$")
plt.ylabel("Log-loss")

# Create a second y-axis for accuracy
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(alpha_values, ac_log_interpolated, label="Training Accuracy", color='#15B01A', alpha=0.8)
ax2.plot(alpha_values, ac_log_interpolated_test, color="#008000", linestyle='--', label="Testing Accuracy", alpha=0.8)
ax2.scatter(0, accuracy_training, label=f'Fitted Training Accuracy: {accuracy_training:.2f}', color='#15B01A')
ax2.set_ylabel("Accuracy")

plt.title("Logistic Regression Loss and Accuracy\nAcross 1D Interpolation of Model Parameters")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

# %% 
# Hamming Distance

bin_rows(log_interpolation)
bin_rows(log_interpolation_test)
hamming_distance_bins(log_interpolation)
hamming_distance_bins(log_interpolation_test)

# %% 

# Standardize the color scale according to hamming distance
hamming_min = min(log_interpolation["hamming_distance"].min(), log_interpolation_test["hamming_distance"].min())
hamming_max = max(log_interpolation["hamming_distance"].max(), log_interpolation_test["hamming_distance"].max())

plt.figure(figsize=(10, 10))
plt.scatter(alpha_values, log_interpolation["loss"], 
            c=log_interpolation["hamming_distance"], 
            cmap="viridis", vmax=hamming_max, vmin=hamming_min, s=20)
plt.xlabel(r"$\alpha$")
plt.ylabel("Log-loss")
plt.title("Training Interpolated 1D Log-loss\n(Hamming Distance)")
plt.colorbar(label="Hamming Distance")  # Add colorbar for reference
plt.grid(True)
plt.savefig("plots/training_interpolated_log_loss.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(alpha_values, log_interpolation_test["loss"], 
            c=log_interpolation_test["hamming_distance"], 
            cmap='viridis', s=20, vmin=hamming_min, vmax=hamming_max)
plt.xlabel(r"$\alpha$")
plt.ylabel("Log-loss")
plt.title("Test Interpolated 1D Log-loss\n(Hamming Distance)")
plt.colorbar(label="Hamming Distance")  # Add colorbar for reference
plt.grid(True)
plt.savefig("plots/test_interpolated_log_loss.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()


# %% 
# Animation! 

fig, ax = plt.subplots(figsize=(10, 10)) 

# Normalize color range for Hamming Distance
norm = mcolors.Normalize(vmin=hamming_min, vmax=hamming_max)
cmap = plt.get_cmap("viridis")

# Initial scatter plot
sc = ax.scatter(alpha_values, log_interpolation["loss"], 
                c=cmap(norm(log_interpolation["hamming_distance"])), s=20, edgecolors='darkgrey', linewidth=0.5)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Log-loss")
ax.set_title("Training Interpolated 1D Log-loss\n(Hamming Distance)\n(Diff between lowest loss and current loss)")
ax.grid(True)

# Add vertical reference line at alpha = 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label="Alpha = 0: Training Fit", alpha=0.5)

# Colorbar
cb = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.set_label("Hamming Distance")

# Unique bins for animation
unique_bins = log_interpolation["bin"].unique()

# Animation update function
def update_training(frame):
    current_bin = unique_bins[frame % len(unique_bins)]
    
    # Identify highlighted points
    is_highlighted = log_interpolation["bin"] == current_bin
    current_colors = [cmap(norm(hd)) if highlight else '#edebeb' 
                      for hd, highlight in zip(log_interpolation["hamming_distance"], is_highlighted)]
    
    # Adjust size: Highlighted points will be larger
    sizes = [50 if highlight else 20 for highlight in is_highlighted]
    
    # Adjust edge color: Highlighted points will have black border, others remain dark grey
    edge_colors = ['black' if highlight else 'darkgrey' for highlight in is_highlighted]
    
    # Update scatter plot properties
    sc.set_sizes(sizes)
    sc.set_color(current_colors)
    sc.set_edgecolors(edge_colors)
    sc.set_linewidths([0.2 if highlight else 0.05 for highlight in is_highlighted])  # Reduced border thickness
    
    return sc,

# Twin axes for Train Accuracy
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(alpha_values, lv_log_interpolated, label="Train Accuracy", color='black', alpha=0.5)

# Create Animation
ani = animation.FuncAnimation(fig, update_training, frames=len(unique_bins), interval=1000, blit=True)

# Save Animation
ani.save("plots/log_training_interpolation.gif", writer="ffmpeg", fps=10)
plt.legend()
plt.show(block=True)

# %% 
# Animation for Testing Interpolation
fig, ax = plt.subplots(figsize=(10, 10))

# Normalize color range for Hamming Distance
norm = mcolors.Normalize(vmin=hamming_min, vmax=hamming_max)
cmap = plt.get_cmap("viridis")

# Initial scatter plot
sc = ax.scatter(alpha_values, log_interpolation_test["loss"], 
                c=cmap(norm(log_interpolation_test["hamming_distance"])), s=20, edgecolors='#edebeb', linewidth=0.05, zorder = 2)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Log-loss")
ax.set_title("Testing Interpolated 1D Log-loss\n(Hamming Distance)\n(Diff between lowest loss and current loss)")
ax.grid(True)

# Colorbar
cb = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.set_label("Hamming Distance")

# Unique bins for animation
unique_bins_test = log_interpolation_test["bin"].unique()

# Animation update function
def update_test(frame):
    current_bin = unique_bins_test[frame % len(unique_bins_test)]
    
    # Identify highlighted points
    is_highlighted = log_interpolation_test["bin"] == current_bin
    current_colors = [cmap(norm(hd)) if highlight else '#edebeb' 
                      for hd, highlight in zip(log_interpolation_test["hamming_distance"], is_highlighted)]
    
    # Adjust size: Highlighted points will be larger
    sizes = [50 if highlight else 20 for highlight in is_highlighted]
    
    # Adjust edge color: Highlighted points will have black border, others remain dark grey
    edge_colors = ['black' if highlight else 'darkgrey' for highlight in is_highlighted]
    
    # Update scatter plot properties
    sc.set_sizes(sizes)
    sc.set_color(current_colors)
    sc.set_edgecolors(edge_colors)
    sc.set_linewidths([0.2 if highlight else 0.05 for highlight in is_highlighted])  # Reduced border thickness
    
    return sc,

# Twin axes for Test Accuracy
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(alpha_values, lv_log_interpolated_test, label="Test Accuracy", color='black', alpha=0.5)

# Create Animation
ani_test = animation.FuncAnimation(fig, update_test, frames=len(unique_bins_test), interval=1000, blit=True)

# Save Animation
ani_test.save("plots/log_testing_interpolation.gif", writer="ffmpeg", fps=10)
plt.legend()
plt.show(block=True)

# %%

# %% 

# 2D Interpolation
theta_opt_2d = np.hstack([logreg.intercept_, logreg.coef_.ravel()])
delta_2d = np.random.randn(*theta_opt_2d.shape)
eta_2d = np.random.randn(*theta_opt_2d.shape)
alpha_values_2d = np.linspace(-10, 10, 500)
beta_values_2d = np.linspace(-10, 10, 500)
if 0 not in alpha_values_2d:
    alpha_values_2d = np.unique(np.append(alpha_values_2d, 0))  # Ensure alpha = 0 is included
if 0 not in beta_values_2d:
    beta_values_2d = np.unique(np.append(beta_values_2d, 0))  # Ensure beta = 0 is included
theta_star_2d = theta_opt_2d + np.random.normal(scale=2.0, size=theta_opt_2d.shape)  # Randomly selecting theta_star
lv_log_interpolated_2d = []
lv_log_interpolated_test_2d = []
ac_log_interpolated_2d = []
ac_log_interpolated_test_2d = []
log_interpolation_2d = {}
log_interpolation_test_2d = {}
for alpha_2d in alpha_values_2d:
    for beta_2d in beta_values_2d:
        theta_alpha_beta_2d = tau_2d(alpha=alpha_2d, beta=beta_2d, theta_opt=theta_opt_2d, delta=delta_2d, eta=eta_2d, normalize=True)
        log_model_interpolated_2d = LogisticRegression(penalty='none', solver='lbfgs')
        log_model_interpolated_2d.intercept_ = np.array([theta_alpha_beta_2d[0]])
        log_model_interpolated_2d.coef_ = np.array([theta_alpha_beta_2d[1:]])
        log_model_interpolated_2d.classes_ = np.array([0, 1])
        probs_2d = log_model_interpolated_2d.predict_proba(X)[:, 1]
        probs_test_2d = log_model_interpolated_2d.predict_proba(X_test)[:, 1]
        lv_log_interpolated_2d.append(log_loss(y, probs_2d))
        lv_log_interpolated_test_2d.append(log_loss(y_test, probs_test_2d))
        ac_log_interpolated_2d.append((y == (probs_2d >= 0.5).astype(int)).mean())  # Calculate accuracy
        ac_log_interpolated_test_2d.append((y_test == (probs_test_2d >= 0.5).astype(int)).mean())  # Calculate accuracy
        log_interpolation_2d[(alpha_2d, beta_2d)] = {
            "inter": theta_alpha_beta_2d[0],
            "coefficients": theta_alpha_beta_2d[1:],
            "loss": log_loss(y, probs_2d),
            "risk_predictions": probs_2d,
            "binary_predictions": (probs_2d >= 0.5).astype(int)  # Threshold = 0.5
        }
        log_interpolation_test_2d[(alpha_2d, beta_2d)] = {
            "inter": theta_alpha_beta_2d[0],
            "coefficients": theta_alpha_beta_2d[1:],
            "loss": log_loss(y_test, probs_test_2d),
            "risk_predictions": probs_test_2d,
            "binary_predictions": (probs_test_2d >= 0.5).astype(int)  # Threshold = 0.5
        }
    if alpha_values_2d.tolist().index(alpha_2d) % 10 == 0:
        print(f"Completed processing alpha_2d = {alpha_2d}")

log_interpolation_2d = pd.DataFrame.from_dict(log_interpolation_2d, orient='index')
log_interpolation_test_2d = pd.DataFrame.from_dict(log_interpolation_test_2d, orient='index')

# %% 
# 3D Plot of Log-loss Across 2D Interpolation
from mpl_toolkits.mplot3d import Axes3D

# Reshape lv_log_interpolated_2d to match the grid of alpha and beta values
alpha_grid, beta_grid = np.meshgrid(alpha_values_2d, beta_values_2d)
loss_grid = np.array(lv_log_interpolated_2d).reshape(len(alpha_values_2d), len(beta_values_2d))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(alpha_grid, beta_grid, loss_grid, cmap='RdPu', edgecolor='none', alpha=0.8)

# Add labels and title
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_zlabel("Log-loss")
ax.set_title("3D Plot of Log-loss Across 2D Interpolation")

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Log-loss")
ax.view_init(elev=20, azim=200)

plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 8))

# Create a filled contour plot
contour = ax.contourf(alpha_grid, beta_grid, loss_grid, levels=50, cmap='RdPu')

# Add contour lines
contour_lines = ax.contour(alpha_grid, beta_grid, loss_grid, levels=10, colors='black', linewidths=0.5)

# Add labels and title
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Contour Plot of Log-loss Across 2D Interpolation")

# Add color bar
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Log-loss")

plt.show()

# %%
# Hamming Distance 
# Hamming Distance

bin_rows(log_interpolation_2d)
bin_rows(log_interpolation_test_2d)
hamming_distance_bins(log_interpolation_2d)
hamming_distance_bins(log_interpolation_test_2d)