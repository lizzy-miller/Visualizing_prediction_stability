# %% 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import ThreeLayerNN, ThreeLayerNNwithSkip
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import random

os.chdir('/Users/lizzy/Desktop/Visualizing_prediction_stability')

# Load Data
data = pd.read_csv('data/log_synthetic_data.csv')
data_ = data

data.head(10)
# %%
# Preprocess Data
# Stratify based on y
test_size = 10000
train_size = 500 
data, test_data = train_test_split(data, test_size = test_size, stratify = data['y'], random_state = 0)

# Sample again to define Training 
train_data, _ = train_test_split(data, train_size=train_size, stratify = data['y'], random_state = 0)
# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[['x', 'z']])
y_train = train_data['y'].values
X_test = scaler.transform(test_data[['x', 'z']])
y_test = test_data['y'].values

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

# %%
# Train Neural Network
lr = 0.01
num_epochs = 1000
model = ThreeLayerNNwithSkip(input_size=2, hidden_size=25, output_size=1)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in the model: {num_params}")

criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    train_outputs_prob = model(X_train_tensor)
    reference_train_loss_predictions = criterion(train_outputs_prob, y_train_tensor)
    train_outputs_binary = (train_outputs_prob >= 0.5).float()
    train_accuracy = (train_outputs_binary == y_train_tensor).float().mean().item()
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    reference_train_loss_binary = criterion(train_outputs_binary, y_train_tensor)
    print(f"Final Train Loss: {reference_train_loss_predictions.item()}")
    test_outputs = model(X_test_tensor)
    reference_test_loss = criterion(test_outputs, y_test_tensor)
    test_accuracy = ((test_outputs >= 0.5).float() == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {reference_test_loss.item()}")
    

# %%
from utils import tau_1d, tau_2d

# Extract model parameters
theta_opt = np.hstack([param.detach().numpy().ravel() for param in model.parameters()])
delta = np.random.randn(*theta_opt.shape)
eta = np.random.randn(*theta_opt.shape)

alpha_values = np.linspace(-20, 20, 1000)
theta_star = theta_opt + np.random.normal(scale=2.0, size=theta_opt.shape)  # randomly selecting theta_star

# %%
lv_nn_interpolated = []
lv_nn_interpolated_test = []
ac_nn_interpolated = []
ac_nn_interpolated_test = []
nn_interpolation = {}
nn_interpolation_test = {}

for alpha in alpha_values:
    theta_alpha = tau_1d(alpha=alpha, theta=theta_opt, theta_star=theta_star, normalize=True)
    # Update model parameters
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param.data = torch.tensor(theta_alpha[start:end].reshape(param.shape), dtype=torch.float)
        start = end

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor).numpy()
        test_outputs = model(X_test_tensor).numpy()
        train_loss = nn.BCELoss()(torch.tensor(train_outputs), y_train_tensor).item()
        test_loss = nn.BCELoss()(torch.tensor(test_outputs), y_test_tensor).item()
        train_accuracy = ((train_outputs >= 0.5).astype(int) == y_train).mean()
        test_accuracy = ((test_outputs >= 0.5).astype(int) == y_test).mean()
    

    lv_nn_interpolated.append(train_loss)
    lv_nn_interpolated_test.append(test_loss)
    ac_nn_interpolated.append(train_accuracy)
    ac_nn_interpolated_test.append(test_accuracy)
    nn_interpolation[alpha] = {
        "inter": theta_alpha[0],
        "coefficients": theta_alpha[1:],
        "loss": train_loss,
        "risk_predictions": train_outputs,
        "binary_predictions": (train_outputs >= 0.5).astype(int)  # threshold = 0.5
    }
    nn_interpolation_test[alpha] = {
        "inter": theta_alpha[0],
        "coefficients": theta_alpha,
        "loss": test_loss,
        "test_predictions": test_outputs,
        "binary_predictions": (test_outputs >= 0.5).astype(int)  # threshold = 0.5
    }

nn_interpolation = pd.DataFrame.from_dict(nn_interpolation, orient='index')
nn_interpolation_test = pd.DataFrame.from_dict(nn_interpolation_test, orient='index')

# Ensure alpha_values are set as the index for nn_interpolation and nn_interpolation_test
nn_interpolation.index = alpha_values
nn_interpolation_test.index = alpha_values

# Create a dataframe to store the estimated risks
#binary_preds_train_dict = {}
#binary_preds_test_dict = {}

#for alpha in alpha_values:
    #binary_preds_train_dict[f'risk_alpha_{alpha}'] = nn_interpolation.loc[alpha, 'binary_predictions']
    #binary_preds_test_dict[f'test_risk_alpha_{alpha}'] = nn_interpolation_test.loc[alpha, 'binary_predictions']

#binary_preds_train = pd.DataFrame(binary_preds_train_dict, index=train_data.index)
#binary_preds_test = pd.DataFrame(binary_preds_test_dict, index=test_data.index)

# %% 
# Loss and Accuracy Plot for Neural Network Interpolation
plt.figure(figsize=(10, 10))

# First y-axis for loss
plt.plot(alpha_values, lv_nn_interpolated, label="Training NN Loss", color="#FF0000")
plt.plot(alpha_values, lv_nn_interpolated_test, color="#E50000", linestyle='--', label="Testing NN Loss")

#plt.axhline(y=optimal_loss_nn_training, color='black', linestyle='--', label=f'Optimal NN Loss Training: {optimal_loss_nn_training:.2f}')
# plt.axhline(y=optimal_loss_nn_test, color='grey', linestyle='--', 
#             label=f'Optimal NN Loss Test: {optimal_loss_nn_test:.4f}')

#plt.scatter(0, loss_nn_training, label=f'Fitted NN Loss Training: {loss_nn_training:.2f}', color='#FF0000')
# plt.scatter(0, loss_nn_testing, label=f'Fitted NN Loss Test: {loss_nn_testing:.2f}', color='#E50000')

plt.xlabel(r"$\alpha$")
plt.ylabel("Loss")

# Second y-axis for accuracy
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(alpha_values, ac_nn_interpolated, label="Training NN Accuracy", color='#15B01A', alpha=0.8)
ax2.plot(alpha_values, ac_nn_interpolated_test, color="#008000", linestyle='--', label="Testing NN Accuracy", alpha=0.8)
#ax2.scatter(0, accuracy_nn_training, label=f'Fitted Training Accuracy: {accuracy_nn_training:.2f}', color='#15B01A')
ax2.set_ylabel("Accuracy")

plt.title("Neural Network Loss and Accuracy\nAcross 1D Interpolation of Model Parameters")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
# plt.grid(True)
plt.savefig("plots/neural_network_loss_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
from utils import bin_rows, hamming_distance_bins
bin_rows(nn_interpolation, coef_threshold=0.01, index_threshold=0.01)
bin_rows(nn_interpolation_test, coef_threshold=0.01, index_threshold=0.01)
hamming_distance_bins(nn_interpolation)
hamming_distance_bins(nn_interpolation_test)
# %% 
# Standardize the color scale according to hamming distance
hamming_min_nn = min(nn_interpolation["hamming_distance"].min(), nn_interpolation_test["hamming_distance"].min())
hamming_max_nn = max(nn_interpolation["hamming_distance"].max(), nn_interpolation_test["hamming_distance"].max())

plt.figure(figsize=(10, 10))
plt.scatter(alpha_values, nn_interpolation["loss"], 
            c=nn_interpolation["hamming_distance"], 
            cmap="viridis", vmin=hamming_min_nn, vmax=hamming_max_nn, s=20)
plt.xlabel(r"$\alpha$")
plt.ylabel("BCELoss")
plt.title("Training Interpolated 1D BCELoss\n(Hamming Distance)")
plt.colorbar(label="Hamming Distance")  # Add colorbar for reference
plt.grid(True)
plt.savefig("plots/training_interpolated_nn_loss.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(alpha_values, nn_interpolation_test["loss"], 
            c=nn_interpolation_test["hamming_distance"], 
            cmap='viridis', vmin=hamming_min_nn, vmax=hamming_max_nn, s=20)
plt.xlabel(r"$\alpha$")
plt.ylabel("BCELoss")
plt.title("Test Interpolated 1D BCELoss\n(Hamming Distance)")
plt.colorbar(label="Hamming Distance")  # Add colorbar for reference
plt.grid(True)
plt.savefig("plots/test_interpolated_nn_loss.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()

# %% 
lowest_alpha_training = nn_interpolation["train_loss"].idxmin()
# %%
# Animation! 

# Animation for nn_interpolation
fig, ax = plt.subplots(figsize=(10, 10))

# Normalize color range for Hamming Distance
norm = mcolors.Normalize(vmin=hamming_min_nn, vmax=hamming_max_nn)
cmap = plt.get_cmap("viridis")

# Initial scatter plot
sc = ax.scatter(alpha_values, nn_interpolation["loss"], 
                c=cmap(norm(nn_interpolation["hamming_distance"])), s=20, edgecolors='darkgrey', linewidth=0.5)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("BCELoss")
ax.set_title("Training Interpolated 1D BCELoss\n(Hamming Distance)\n(Diff between lowest loss and current loss)")
ax.grid(True)

# Add vertical reference line at alpha = 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label="Alpha = 0: Training Fit", alpha=0.5)

# Colorbar
cb = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.set_label("Hamming Distance")

# Unique bins for animation
unique_bins = nn_interpolation["bin"].unique()

# Animation update function
def update_training(frame):
    current_bin = unique_bins[frame % len(unique_bins)]
    
    # Identify highlighted points
    is_highlighted = nn_interpolation["bin"] == current_bin
    current_colors = [cmap(norm(hd)) if highlight else '#edebeb' 
                      for hd, highlight in zip(nn_interpolation["hamming_distance"], is_highlighted)]
    
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
ax2.plot(alpha_values, ac_nn_interpolated, label="Train Accuracy", color='black', alpha=0.5)

# Create Animation
ani = animation.FuncAnimation(fig, update_training, frames=len(unique_bins), interval=1000, blit=True)

# Save Animation
ani.save("plots/nn_training_interpolation.gif", writer="ffmpeg", fps=10)
plt.legend()
plt.show(block=True)
# %%
