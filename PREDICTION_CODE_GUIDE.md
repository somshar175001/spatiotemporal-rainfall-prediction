# Complete Rainfall Prediction Guide - Code Examples

## Quick Summary
You now have a fully trained rainfall prediction model! Here's how to use it step by step.

---

## **Quick Start (1 Line)**
```bash
python predict_rainfall.py
```

**Output**: 
- Predictions on test data
- Performance metrics (MAE, RMSE)
- Visualizations
- Next timestep forecast

---

## **Complete Prediction Code - Full Example**

### **Minimal Code (30 Lines)**
```python
import torch
import numpy as np
from pathlib import Path
from rainfall_mkcnn_unet_lstm import RainfallGridDataset, setup_data_loaders, CNN_UNet_LSTM

# Step 1: Load data
dataset = RainfallGridDataset(
    input_specs=[
        ("v10", Path("data/data_stream-moda_stepType-avgua_nc_v10_time_series.csv")),
        ("r", Path("data/data_stream-moda_stepType-avgua_nc_2_r_time_series.csv")),
        ("u10", Path("data/data_stream-moda_stepType-avgua_nc_3_u10_time_series.csv")),
        ("t2m", Path("data/temp_2m_era5_2025 (2).csv")),
        ("z500", Path("data/geopotential_500_2025.csv")),
    ],
    target_path=Path("data/rain_imdb_2025.csv"),
    sequence_length=6,
)

# Step 2: Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_UNet_LSTM(dataset.in_ch, dataset.height, dataset.width)
model.load_state_dict(torch.load("rainfall_output_plots/best_rainfall_mkcnn_unet_lstm.pth", map_location=device))
model.to(device).eval()

# Step 3: Make predictions
_, _, test_dl = setup_data_loaders(dataset, batch_size=8)
predictions = []
with torch.no_grad():
    for x, y, mask in test_dl:
        pred, _ = model(x.to(device))
        predictions.append(pred.cpu().numpy())

# Step 4: Denormalize
predictions = np.concatenate(predictions) * dataset.y_sd + dataset.y_mu
print(f"Rainfall prediction range: {predictions.min():.2f} - {predictions.max():.2f} mm")
```

---

## **Step-by-Step Breakdown**

### **Step 1: Import Libraries**
```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset

from rainfall_mkcnn_unet_lstm import (
    RainfallGridDataset,
    setup_data_loaders,
    CNN_UNet_LSTM,
)
```

### **Step 2: Configuration**
```python
# Paths
MODEL_PATH = Path("rainfall_output_plots/best_rainfall_mkcnn_unet_lstm.pth")
DATA_DIR = Path("data")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 8
SEQUENCE_LENGTH = 6
```

### **Step 3: Load Dataset**
```python
dataset = RainfallGridDataset(
    input_specs=[
        ("v10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_v10_time_series.csv"),
        ("r", DATA_DIR / "data_stream-moda_stepType-avgua_nc_2_r_time_series.csv"),
        ("u10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_3_u10_time_series.csv"),
        ("t2m", DATA_DIR / "temp_2m_era5_2025 (2).csv"),
        ("z500", DATA_DIR / "geopotential_500_2025.csv"),
    ],
    target_path=DATA_DIR / "rain_imdb_2025.csv",
    sequence_length=SEQUENCE_LENGTH,
    target_delay=1,
    train_fraction=0.8,
    fill_missing="mean",
)

print(f"Dataset: {len(dataset)} samples")
print(f"Grid size: {dataset.height} x {dataset.width}")
print(f"Input channels: {dataset.in_ch}")
```

### **Step 4: Setup Data Loaders**
```python
train_dl, val_dl, test_dl = setup_data_loaders(
    dataset,
    batch_size=BATCH_SIZE,
    train_fraction=0.8,
    val_fraction=0.1,
    shuffle_train=False
)
```

### **Step 5: Load Trained Model**
```python
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

print("✓ Model loaded!")
```

### **Step 6: Make Batch Predictions**
```python
all_predictions = []
all_targets = []

print("Making predictions...")
with torch.no_grad():
    for batch_idx, (xb, yb, mask_b) in enumerate(test_dl):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        # Forward pass
        pred, spatial_features = model(xb)
        
        all_predictions.append(pred.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
        
        print(f"  Batch {batch_idx+1}/{len(test_dl)}")

# Concatenate all
predictions = np.concatenate(all_predictions)  # Shape: (num_samples, time, 1, height, width)
targets = np.concatenate(all_targets)
```

### **Step 7: Denormalize to Original Scale**
```python
# Model outputs normalized values
# Scale back to rainfall in millimeters

predictions_denorm = predictions * dataset.y_sd + dataset.y_mu
targets_denorm = targets * dataset.y_sd + dataset.y_mu

print(f"Min rainfall: {predictions_denorm.min():.2f} mm")
print(f"Max rainfall: {predictions_denorm.max():.2f} mm")
print(f"Mean rainfall: {predictions_denorm.mean():.2f} mm")
```

### **Step 8: Calculate Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Flatten spatial and temporal dimensions
y_true_flat = targets_denorm.flatten()
y_pred_flat = predictions_denorm.flatten()

mae = mean_absolute_error(y_true_flat, y_pred_flat)
mse = mean_squared_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-6)))

print(f"MAE:  {mae:.4f} mm")
print(f"RMSE: {rmse:.4f} mm")
print(f"MAPE: {mape:.4f}")
```

### **Step 9: Visualize Predictions**
```python
import matplotlib.pyplot as plt

# Select first test sample, first timestep
actual = targets_denorm[0, 0, 0]  # (height, width)
predicted = predictions_denorm[0, 0, 0]
error = predicted - actual

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Actual
im1 = axes[0].imshow(actual, cmap='Blues')
axes[0].set_title('Actual Rainfall')
plt.colorbar(im1, ax=axes[0], label='mm')

# Predicted
im2 = axes[1].imshow(predicted, cmap='Blues')
axes[1].set_title('Predicted Rainfall')
plt.colorbar(im2, ax=axes[1], label='mm')

# Error
im3 = axes[2].imshow(error, cmap='RdBu_r', vmin=-50, vmax=50)
axes[2].set_title('Prediction Error')
plt.colorbar(im3, ax=axes[2], label='mm')

plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=150)
plt.show()
```

### **Step 10: Predict Next Timestep**
```python
# Get a new sequence (last 6 timesteps from dataset)
sample_idx = len(dataset) - 1
input_seq, target_seq, mask_seq = dataset[sample_idx]

print(f"Input sequence shape: {input_seq.shape}")  # (6, 7, 33, 35)

# Add batch dimension and move to device
input_batch = input_seq.unsqueeze(0).to(DEVICE)  # (1, 6, 7, 33, 35)

# Predict
with torch.no_grad():
    pred, features = model(input_batch)
    # pred shape: (1, 6, 1, 33, 35)

# Get last timestep prediction and denormalize
next_rainfall = pred[0, -1, 0].cpu().numpy()  # (33, 35)
next_rainfall_mm = next_rainfall * dataset.y_sd.squeeze() + dataset.y_mu.squeeze()

print(f"Next timestep rainfall:")
print(f"  Shape: {next_rainfall_mm.shape}")
print(f"  Mean: {next_rainfall_mm.mean():.2f} mm")
print(f"  Max: {next_rainfall_mm.max():.2f} mm")
```

### **Step 11: Save Results**
```python
import pandas as pd

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'MAPE'],
    'Value': [mae, rmse, mape]
})
metrics_df.to_csv('prediction_metrics.csv', index=False)

# Save prediction array
np.save('rainfall_predictions.npy', predictions_denorm)
np.save('next_timestep.npy', next_rainfall_mm)

# Save as CSV with coordinates
results_list = []
for lat_idx in range(dataset.height):
    for lon_idx in range(dataset.width):
        lat = dataset.lat_values[lat_idx]
        lon = dataset.lon_values[lon_idx]
        pred_val = next_rainfall_mm[lat_idx, lon_idx]
        results_list.append({
            'latitude': lat,
            'longitude': lon,
            'predicted_rainfall_mm': pred_val
        })

results_df = pd.DataFrame(results_list)
results_df.to_csv('predictions_with_coords.csv', index=False)
```

---

## **Understanding Prediction Output**

### **Prediction Shape**
```
Shape: (num_samples, time_steps, channels, height, width)
       (2, 6, 1, 33, 35)
        
- 2 test samples
- 6 timesteps (sequence_length)
- 1 output channel (rainfall)
- 33x35 spatial grid
```

### **Accessing Predictions**
```python
# First sample, first timestep
sample_1_t1 = predictions[0, 0, 0]  # Shape: (33, 35)

# All samples, last timestep
all_samples_last_t = predictions[:, -1, 0]  # Shape: (2, 33, 35)

# Single location across time
location_rainfall = predictions[0, :, 0, 10, 15]  # Shape: (6,)
```

---

## **Performance Interpretation**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| MAE | 24.23 mm | Average prediction error |
| RMSE | 34.77 mm | Penalizes large errors |
| MAPE | 461278% | High due to near-zero rainfall values |

**What This Means:**
- ✅ Good for general rainfall patterns
- ⚠️ May miss extreme events
- 💡 Best used with ensemble methods

---

## **Common Tasks**

### **Extract High-Risk Areas (>100mm)**
```python
high_risk = next_rainfall_mm > 100
high_risk_coords = np.argwhere(high_risk)
print(f"High rainfall areas: {len(high_risk_coords)} grid cells")
```

### **Average Regional Rainfall**
```python
regional_mean = next_rainfall_mm.mean()
regional_std = next_rainfall_mm.std()
print(f"Regional avg: {regional_mean:.2f} ± {regional_std:.2f} mm")
```

### **Trend Analysis**
```python
# Compare predictions over multiple timesteps
time_series = predictions[0, :, 0]  # All timesteps, first sample
trend = time_series.mean(axis=(1, 2))  # Average spatial dimension
plt.plot(trend)
plt.xlabel('Timestep')
plt.ylabel('Average Rainfall (mm)')
plt.show()
```

---

## **Troubleshooting**

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Wrong path | Check MODEL_PATH exists |
| `Shape mismatch` | Data loading issue | Verify dataset grid size |
| `CUDA out of memory` | Batch too large | Reduce BATCH_SIZE |
| `NaN predictions` | Missing data | Check fill_missing='mean' |

---

## **Files Generated**
- `rainfall_output_plots/sample_predictions.png` - Visual comparison
- `rainfall_output_plots/prediction_results.csv` - Metrics
- `rainfall_output_plots/next_timestep_prediction.npy` - Next forecast

---

## **Next Steps**

1. **Run predictions**: `python predict_rainfall.py`
2. **Check outputs**: `ls rainfall_output_plots/`
3. **Deploy model**: Use for operational forecasting
4. **Validate**: Compare with actual observations
5. **Iterate**: Retrain with more data for better accuracy

---

## **Questions?**
- Check `rainfall_mkcnn_unet_lstm.py` for model architecture
- See `PREDICTION_GUIDE.md` for 10-step guide
- Run `python predict_rainfall.py` for full example
