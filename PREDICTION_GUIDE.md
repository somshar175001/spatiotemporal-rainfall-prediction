# Rainfall Prediction Guide - Step by Step

## Quick Start
```bash
python predict_rainfall.py
```

---

## 10 Steps to Make Predictions

### **STEP 1: Configuration**
```python
from pathlib import Path
import torch

MODEL_PATH = Path("rainfall_output_plots/best_rainfall_mkcnn_unet_lstm.pth")
DATA_DIR = Path("data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### **STEP 2: Load Dataset**
```python
from rainfall_mkcnn_unet_lstm import RainfallGridDataset

dataset = RainfallGridDataset(
    input_specs=[
        ("v10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_v10_time_series.csv"),
        ("r", DATA_DIR / "data_stream-moda_stepType-avgua_nc_2_r_time_series.csv"),
        ("u10", DATA_DIR / "data_stream-moda_stepType-avgua_nc_3_u10_time_series.csv"),
        ("t2m", DATA_DIR / "temp_2m_era5_2025 (2).csv"),
        ("z500", DATA_DIR / "geopotential_500_2025.csv"),
    ],
    target_path=DATA_DIR / "rain_imdb_2025.csv",
    sequence_length=6,
    target_delay=1,
)
```

### **STEP 3: Setup Data Loaders**
```python
from rainfall_mkcnn_unet_lstm import setup_data_loaders

train_dl, val_dl, test_dl = setup_data_loaders(
    dataset, batch_size=8, train_fraction=0.8
)
```

### **STEP 4: Load Trained Model**
```python
from rainfall_mkcnn_unet_lstm import MKCNNUNETLSTM

model = MKCNNUNETLSTM(
    in_channels=dataset.in_ch,
    out_channels=1,
    sequence_length=6,
    hidden_dim=96,
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
```

### **STEP 5: Make Predictions**
```python
import numpy as np

all_predictions = []
all_targets = []

with torch.no_grad():
    for xb, yb, mask_b in test_dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        pred, _ = model(xb)  # Forward pass
        
        all_predictions.append(pred.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

predictions = np.concatenate(all_predictions, axis=0)
targets = np.concatenate(all_targets, axis=0)
```

### **STEP 6: Denormalize Predictions**
```python
# Convert from normalized scale back to mm
predictions_denorm = predictions * dataset.y_sd + dataset.y_mu
targets_denorm = targets * dataset.y_sd + dataset.y_mu

print(f"Min rainfall: {predictions_denorm.min():.2f} mm")
print(f"Max rainfall: {predictions_denorm.max():.2f} mm")
```

### **STEP 7: Calculate Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(targets_denorm.flatten(), predictions_denorm.flatten())
rmse = np.sqrt(mean_squared_error(targets_denorm.flatten(), predictions_denorm.flatten()))

print(f"MAE: {mae:.4f} mm")
print(f"RMSE: {rmse:.4f} mm")
```

### **STEP 8: Visualize Results**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Show first sample
i = 0
axes[0].imshow(targets_denorm[i, 0], cmap='Blues')
axes[0].set_title('Actual Rainfall')

axes[1].imshow(predictions_denorm[i, 0], cmap='Blues')
axes[1].set_title('Predicted Rainfall')

axes[2].imshow(predictions_denorm[i, 0] - targets_denorm[i, 0], cmap='RdBu_r')
axes[2].set_title('Prediction Error')

plt.tight_layout()
plt.savefig('prediction_comparison.png')
```

### **STEP 9: Predict Next Time Step**
```python
# Use the last sequence from dataset
last_sample_idx = len(dataset) - 1
last_x, last_y, last_mask = dataset[last_sample_idx]

with torch.no_grad():
    input_batch = last_x.unsqueeze(0).to(DEVICE)
    prediction, features = model(input_batch)

# Denormalize to get rainfall in mm
pred_denorm = prediction.squeeze().cpu().numpy() * dataset.y_sd.squeeze() + dataset.y_mu.squeeze()

print(f"Next timestep rainfall prediction:")
print(f"Shape: {pred_denorm.shape}")
print(f"Mean: {pred_denorm.mean():.2f} mm")
```

### **STEP 10: Save Results**
```python
import pandas as pd

results = {
    'MAE': [mae],
    'RMSE': [rmse],
    'Mean_Predicted_mm': [predictions_denorm.mean()],
    'Mean_Actual_mm': [targets_denorm.mean()],
}

df = pd.DataFrame(results)
df.to_csv('prediction_results.csv', index=False)

# Save prediction array
np.save('next_rainfall_prediction.npy', pred_denorm)
```

---

## Understanding the Prediction Output

### **Prediction Shape**
- `predictions.shape`: `(num_samples, 1, height, width)`
  - `num_samples`: Number of test samples
  - `1`: Output channel (single rainfall variable)
  - `height, width`: Spatial grid dimensions

### **Example Values**
```
Min rainfall:    0.00 mm
Max rainfall:  150.45 mm
Mean rainfall:   45.68 mm
Prediction Error: ±54.61 mm (RMSE)
```

### **How to Use Results**
1. **Flood Warning**: Flag regions where prediction > 100mm
2. **Agriculture**: Use prediction for irrigation planning
3. **Drought Monitoring**: Track low rainfall regions
4. **Climate Analysis**: Analyze spatial/temporal patterns

---

## Practical Applications

### **Real-time Forecasting**
```python
# Load latest meteorological data
# Run prediction
# Issue warnings if rainfall > threshold
```

### **Ensemble Predictions**
```python
# Load multiple models trained with different parameters
# Average predictions for robustness
# Calculate uncertainty bands
```

### **Time Series Forecasting**
```python
# Use predicted rainfall as input for next timestep
# Chain predictions for multi-step forecast
# Note: Errors compound over longer horizons
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check `MODEL_PATH` exists |
| CUDA out of memory | Reduce `BATCH_SIZE` |
| Shape mismatch | Verify dataset grid size matches model |
| NaN predictions | Check for missing values in input data |

---

## Performance Expectations

Based on test set:
- **MAE**: 45.69 mm (average error)
- **RMSE**: 54.61 mm (penalizes large errors)
- **Best for**: Regional/medium-range forecasts
- **Use with caution**: Extreme rainfall events

---

## Next Steps

✅ Run: `python predict_rainfall.py`  
✅ Check outputs in: `rainfall_output_plots/`  
✅ Deploy model for operational forecasting  
✅ Validate against real observations  
