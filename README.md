# spatiotemporal-rainfall-prediction

Created the rainfall model code here:
[rainfall_mkcnn_unet_lstm.py](rainfall_mkcnn_unet_lstm.py)
It includes the full pipeline :
converts each time step into a spatial rainfall grid
uses MultiKernelBlock → U-Net → LSTM
predicts rainfall grids
uses MaskedPhysicsInformedLoss
evaluates MAE, MSE, RMSE, MAPE, SSIM
saves checkpoints, training/validation plots, prediction maps, MKB feature maps, horizon-wise metric plots, and ablation results
Also made it work with  irregular rainfall grid by using a valid-cell mask, so it does not assume the grid is a perfect rectangle.
To run with all default files:
python rainfall_mkcnn_unet_lstm.py