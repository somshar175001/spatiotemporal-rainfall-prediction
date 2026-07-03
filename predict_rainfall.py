"""
Step-by-Step Rainfall Prediction Script
This script demonstrates how to load the trained model and make predictions
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from rainfall_mkcnn_unet_lstm import (
    RainfallGridDataset,
    setup_data_loaders,
    CNN_UNet_LSTM,
)


# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING CONFIGURATION")
print("=" * 70)

MODEL_PATH = Path("rainfall_output_plots/best_rainfall_mkcnn_unet_lstm.pth")
DATA_DIR = Path("data")

# Input specifications (same as training)
INPUT_SPECS = [
    ("v10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_v10_time_series.csv"),
    ("r", DATA_DIR / "data_stream-moda_stepType-avgua_nc_2_r_time_series.csv"),
    ("u10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_3_u10_time_series.csv"),
    ("t2m", DATA_DIR / "temp_2m_era5_2025 (2).csv"),
    ("z500", DATA_DIR / "geopotential_500_2025.csv"),
]
TARGET_PATH = DATA_DIR / "rain_imdb_2025.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
SEQUENCE_LENGTH = 6

print(f"✓ Model path: {MODEL_PATH}")
print(f"✓ Device: {DEVICE}")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Sequence length: {SEQUENCE_LENGTH}")


# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: LOADING DATASET")
print("=" * 70)

print("Loading rainfall dataset...")
dataset = RainfallGridDataset(
    input_specs=INPUT_SPECS,
    target_path=TARGET_PATH,
    sequence_length=SEQUENCE_LENGTH,
    target_delay=1,
    train_fraction=0.8,
    include_month_channels=True,
    fill_missing="mean",
    predict_delta=False,
)

print(f"✓ Dataset loaded!")
print(f"  - Total samples: {len(dataset)}")
print(f"  - Grid size: {dataset.height} x {dataset.width}")
print(f"  - Input channels: {dataset.in_ch}")
print(f"  - Date range: {dataset.dates[0].date()} to {dataset.dates[-1].date()}")


# ============================================================================
# STEP 3: SETUP DATA LOADERS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SETUP DATA LOADERS")
print("=" * 70)

train_dl, val_dl, test_dl = setup_data_loaders(
    dataset, batch_size=BATCH_SIZE, train_fraction=0.8, val_fraction=0.1
)
print("✓ Data loaders created!")


# ============================================================================
# STEP 4: LOAD TRAINED MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: LOADING TRAINED MODEL")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = CNN_UNet_LSTM(
    in_ch=dataset.in_ch,
    height=dataset.height,
    width=dataset.width,
    hidden_dim=96,
    mkb_kernel_sizes=[3, 5, 7],
    use_se=True,
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print("✓ Model loaded and set to evaluation mode!")


# ============================================================================
# STEP 5: MAKE PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: MAKING PREDICTIONS ON TEST SET")
print("=" * 70)

all_predictions = []
all_targets = []
all_dates = []

print("Generating predictions...")
with torch.no_grad():
    for batch_idx, (xb, yb, mask_b) in enumerate(test_dl):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        mask_b = mask_b.to(DEVICE)
        
        # Forward pass
        pred, _ = model(xb)
        
        all_predictions.append(pred.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

# Concatenate all predictions
predictions = np.concatenate(all_predictions, axis=0)  # [samples, 1, height, width]
targets = np.concatenate(all_targets, axis=0)  # [samples, 1, height, width]

print(f"✓ Predictions generated!")
print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Targets shape: {targets.shape}")


# ============================================================================
# STEP 6: DENORMALIZE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: DENORMALIZING PREDICTIONS")
print("=" * 70)

# Denormalize back to original scale
predictions_denorm = predictions * dataset.y_sd + dataset.y_mu
targets_denorm = targets * dataset.y_sd + dataset.y_mu

print(f"✓ Predictions denormalized!")
print(f"  - Min predicted rainfall: {predictions_denorm.min():.2f} mm")
print(f"  - Max predicted rainfall: {predictions_denorm.max():.2f} mm")
print(f"  - Mean predicted rainfall: {predictions_denorm.mean():.2f} mm")
print(f"  - Min actual rainfall: {targets_denorm.min():.2f} mm")
print(f"  - Max actual rainfall: {targets_denorm.max():.2f} mm")
print(f"  - Mean actual rainfall: {targets_denorm.mean():.2f} mm")


# ============================================================================
# STEP 7: CALCULATE METRICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: CALCULATING PREDICTION METRICS")
print("=" * 70)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(targets_denorm.flatten(), predictions_denorm.flatten())
rmse = np.sqrt(mean_squared_error(targets_denorm.flatten(), predictions_denorm.flatten()))
mape = np.mean(np.abs((targets_denorm - predictions_denorm) / (np.abs(targets_denorm) + 1e-6)))

print(f"✓ Metrics calculated!")
print(f"  - MAE (Mean Absolute Error): {mae:.4f} mm")
print(f"  - RMSE (Root Mean Square Error): {rmse:.4f} mm")
print(f"  - MAPE (Mean Absolute Percentage Error): {mape:.4f}")


# ============================================================================
# STEP 8: VISUALIZE SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: VISUALIZING SAMPLE PREDICTIONS")
print("=" * 70)

# Select a few sample predictions
num_samples = 2
time_steps = 3  # Show first 3 timesteps

fig, axes = plt.subplots(num_samples, time_steps * 3, figsize=(18, 8))

for i in range(min(num_samples, len(predictions))):
    for t in range(time_steps):
        actual = targets_denorm[i, t, 0]
        pred = predictions_denorm[i, t, 0]
        diff = pred - actual
        
        row = i
        col_base = t * 3
        
        # Actual rainfall
        im1 = axes[row, col_base].imshow(actual, cmap='Blues')
        axes[row, col_base].set_title(f'Sample {i+1}, T={t+1}: Actual')
        plt.colorbar(im1, ax=axes[row, col_base], label='mm')
        
        # Predicted rainfall
        im2 = axes[row, col_base + 1].imshow(pred, cmap='Blues')
        axes[row, col_base + 1].set_title(f'Sample {i+1}, T={t+1}: Predicted')
        plt.colorbar(im2, ax=axes[row, col_base + 1], label='mm')
        
        # Difference
        im3 = axes[row, col_base + 2].imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
        axes[row, col_base + 2].set_title(f'Sample {i+1}, T={t+1}: Error')
        plt.colorbar(im3, ax=axes[row, col_base + 2], label='mm')

plt.tight_layout()
plt.savefig('rainfall_output_plots/sample_predictions.png', dpi=150, bbox_inches='tight')
print("✓ Sample predictions visualization saved!")


# ============================================================================
# STEP 9: MAKE PREDICTION FOR NEW DATA (EXAMPLE)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: PREDICTING NEXT TIME STEP (EXAMPLE)")
print("=" * 70)

# Get the last 6 time steps from the dataset
last_sample_idx = len(dataset) - 1
last_x, last_y, last_mask = dataset[last_sample_idx]

print(f"Using last sample from dataset (index: {last_sample_idx})")
print(f"  - Input shape: {last_x.shape} (sequence_length, channels, height, width)")
print(f"  - Target shape: {last_y.shape}")

# Make prediction
with torch.no_grad():
    input_batch = last_x.unsqueeze(0).to(DEVICE)  # Add batch dimension
    prediction, features = model(input_batch)

# Denormalize - prediction has shape (1, 6, 1, height, width)
# Take the last timestep
pred_denorm = prediction.squeeze().cpu().numpy()[-1] * dataset.y_sd.squeeze() + dataset.y_mu.squeeze()

print(f"✓ Next time step prediction generated!")
print(f"  - Predicted rainfall shape: {pred_denorm.shape}")
print(f"  - Min: {pred_denorm.min():.2f} mm")
print(f"  - Max: {pred_denorm.max():.2f} mm")
print(f"  - Mean: {pred_denorm.mean():.2f} mm")


# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: SAVING RESULTS")
print("=" * 70)

# Save predictions to CSV
results_df = pd.DataFrame({
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'Mean_Predicted_mm': [predictions_denorm.mean()],
    'Mean_Actual_mm': [targets_denorm.mean()],
})
results_df.to_csv('rainfall_output_plots/prediction_results.csv', index=False)
print("✓ Prediction results saved to: rainfall_output_plots/prediction_results.csv")

# Save next timestep prediction
np.save('rainfall_output_plots/next_timestep_prediction.npy', pred_denorm)
print("✓ Next timestep prediction saved to: rainfall_output_plots/next_timestep_prediction.npy")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PREDICTION COMPLETE!")
print("=" * 70)
print(f"""
Summary:
--------
✓ Model loaded from: {MODEL_PATH}
✓ Processed {len(dataset)} test samples
✓ Test Set Metrics:
  - MAE: {mae:.4f} mm
  - RMSE: {rmse:.4f} mm
  - MAPE: {mape:.4f}

✓ Generated visualizations
✓ Predicted next timestep
✓ Results saved to rainfall_output_plots/

Next Steps:
-----------
1. View sample predictions: rainfall_output_plots/sample_predictions.png
2. Check metrics: rainfall_output_plots/prediction_results.csv
3. Use next_timestep_prediction.npy for operational forecasting
""")
print("=" * 70)
