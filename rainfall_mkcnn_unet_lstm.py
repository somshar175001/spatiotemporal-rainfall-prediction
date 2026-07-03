import argparse
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


DEFAULT_DATA_DIR = Path("/Users/ishitasharma/Desktop/EveryThing/Somya")
DEFAULT_INPUTS = [
    ("v10", DEFAULT_DATA_DIR / "data_stream-moda_stepType-avgua_nc_v10_time_series.csv"),
    ("r", DEFAULT_DATA_DIR / "data_stream-moda_stepType-avgua_nc_2_r_time_series.csv"),
    ("u10", DEFAULT_DATA_DIR / "data_stream-moda_stepType-avgua_nc_3_u10_time_series.csv"),
    ("t2m", DEFAULT_DATA_DIR / "temp_2m_era5_2025 (2).csv"),
    ("z500", DEFAULT_DATA_DIR / "geopotential_500_2025.csv"),
]
DEFAULT_TARGET = DEFAULT_DATA_DIR / "rain_imdb_2025.csv"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_date_column(col):
    text = str(col).strip()
    if text.lower() in {"pixel_id", "latitude", "longitude", "lat", "lon"}:
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(parsed.year, parsed.month, 1)

    clean = text.replace("_", "-").replace("/", "-")
    parts = clean.split("-")
    if len(parts) == 2 and parts[0].lower()[:3] in MONTHS:
        month = MONTHS[parts[0].lower()[:3]]
        year = int(parts[1])
        if year < 100:
            year += 1900 if year >= 70 else 2000
        return pd.Timestamp(year, month, 1)

    return None


def read_grid_time_series(path, name=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    lat_col = "Latitude" if "Latitude" in df.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in df.columns else "longitude"
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"{path.name} must contain Latitude/Longitude or latitude/longitude columns.")

    date_cols = []
    date_map = {}
    for col in df.columns:
        dt = parse_date_column(col)
        if dt is not None:
            date_cols.append(col)
            date_map[col] = dt

    if not date_cols:
        raise ValueError(f"{path.name} does not contain recognizable monthly/date columns.")

    work = df[[lat_col, lon_col] + date_cols].copy()
    work = work.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    work["latitude"] = work["latitude"].astype(float).round(6)
    work["longitude"] = work["longitude"].astype(float).round(6)

    # Rename date columns to normalized Timestamp objects; duplicate dates are averaged.
    rename = {col: date_map[col] for col in date_cols}
    work = work.rename(columns=rename)
    grouped = work.groupby(["latitude", "longitude"], as_index=False).mean(numeric_only=True)
    date_columns = sorted([c for c in grouped.columns if isinstance(c, pd.Timestamp)])
    grouped = grouped[["latitude", "longitude"] + date_columns]

    return {
        "name": name or path.stem,
        "path": path,
        "frame": grouped,
        "dates": date_columns,
    }


def build_spatial_stack(frame, dates, target_coords, lat_values, lon_values):
    merged = target_coords.merge(frame, on=["latitude", "longitude"], how="left")
    h, w = len(lat_values), len(lon_values)
    lat_to_i = {v: i for i, v in enumerate(lat_values)}
    lon_to_j = {v: j for j, v in enumerate(lon_values)}
    out = np.full((len(dates), h, w), np.nan, dtype=np.float32)

    for _, row in merged.iterrows():
        i = lat_to_i[float(row["latitude"])]
        j = lon_to_j[float(row["longitude"])]
        vals = row[dates].to_numpy(dtype=np.float32)
        out[:, i, j] = vals

    return out


class RainfallGridDataset(Dataset):
    def __init__(
        self,
        input_specs,
        target_path,
        sequence_length=6,
        target_delay=1,
        train_fraction=0.8,
        include_month_channels=True,
        fill_missing="mean",
        predict_delta=False,
    ):
        self.sequence_length = int(sequence_length)
        self.target_delay = int(target_delay)
        self.include_month_channels = bool(include_month_channels)
        self.predict_delta = bool(predict_delta)

        target = read_grid_time_series(target_path, "rainfall")
        inputs = [read_grid_time_series(path, name) for name, path in input_specs]

        common_dates = set(target["dates"])
        for item in inputs:
            common_dates &= set(item["dates"])
        self.dates = sorted(common_dates)
        if len(self.dates) < self.sequence_length + self.target_delay + 1:
            raise ValueError(
                "Not enough common dates across all files. "
                f"Common dates={len(self.dates)}, needed at least {self.sequence_length + self.target_delay + 1}. "
                "Check sparse inputs such as annual-only humidity, or remove that input."
            )

        print(f"Using {len(self.dates)} common dates: {self.dates[0].date()} to {self.dates[-1].date()}")
        for item in inputs:
            if len(item["dates"]) != len(self.dates):
                print(
                    f"Warning: {item['name']} has {len(item['dates'])} dates; "
                    f"only {len(self.dates)} overlap with all files."
                )

        coords = target["frame"][["latitude", "longitude"]].copy()
        coords = coords.drop_duplicates().sort_values(["latitude", "longitude"]).reset_index(drop=True)
        self.lat_values = sorted(coords["latitude"].unique(), reverse=True)
        self.lon_values = sorted(coords["longitude"].unique())
        self.height = len(self.lat_values)
        self.width = len(self.lon_values)

        valid_mask = np.zeros((self.height, self.width), dtype=np.float32)
        lat_to_i = {v: i for i, v in enumerate(self.lat_values)}
        lon_to_j = {v: j for j, v in enumerate(self.lon_values)}
        for _, row in coords.iterrows():
            valid_mask[lat_to_i[float(row["latitude"])]][lon_to_j[float(row["longitude"])]] = 1.0
        self.valid_mask = valid_mask[None, :, :].astype(np.float32)

        x_channels = []
        self.input_names = []
        for item in inputs:
            arr = build_spatial_stack(item["frame"], self.dates, coords, self.lat_values, self.lon_values)
            x_channels.append(arr)
            self.input_names.append(item["name"])

        x = np.stack(x_channels, axis=1).astype(np.float32)  # [time, channel, height, width]
        y = build_spatial_stack(target["frame"], self.dates, coords, self.lat_values, self.lon_values)[:, None]

        x = self._fill_missing(x, fill_missing)
        y = self._fill_missing(y, fill_missing)
        self.raw_target = y.copy()

        train_frame_count = max(self.sequence_length + 1, int(len(self.dates) * float(train_fraction)))
        train_x = x[:train_frame_count]
        train_y = y[:train_frame_count]

        self.x_mu = np.nanmean(train_x, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self.x_sd = (np.nanstd(train_x, axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
        self.y_mu = np.nanmean(train_y, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self.y_sd = (np.nanstd(train_y, axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)

        x = (x - self.x_mu) / self.x_sd

        if self.predict_delta:
            y_delta = np.zeros_like(y)
            y_delta[1:] = y[1:] - y[:-1]
            y = y_delta

        y = (y - self.y_mu) / self.y_sd

        if self.include_month_channels:
            month_sin = []
            month_cos = []
            for dt in self.dates:
                angle = 2.0 * np.pi * ((dt.month - 1) / 12.0)
                month_sin.append(np.full((self.height, self.width), np.sin(angle), dtype=np.float32))
                month_cos.append(np.full((self.height, self.width), np.cos(angle), dtype=np.float32))
            x = np.concatenate(
                [x, np.stack(month_sin)[:, None], np.stack(month_cos)[:, None]],
                axis=1,
            )
            self.input_names += ["month_sin", "month_cos"]

        self.in_ch = int(x.shape[1])
        self.X_data = []
        self.Y_data = []
        max_idx = len(self.dates) - self.target_delay
        for i in range(self.sequence_length, max_idx):
            self.X_data.append(x[i - self.sequence_length : i])
            self.Y_data.append(y[i - self.sequence_length + self.target_delay : i + self.target_delay])

        self.X_data = np.stack(self.X_data).astype(np.float32)
        self.Y_data = np.stack(self.Y_data).astype(np.float32)
        self.mask_data = np.broadcast_to(
            self.valid_mask[None, None], (len(self.X_data), self.sequence_length, 1, self.height, self.width)
        ).astype(np.float32)

        print(f"Grid: {self.height} x {self.width}; valid rainfall cells: {int(valid_mask.sum())}")
        print(f"Inputs: {self.input_names}")
        print(f"Dataset X shape: {self.X_data.shape}")
        print(f"Dataset Y shape: {self.Y_data.shape}")

    @staticmethod
    def _fill_missing(arr, mode):
        arr = arr.astype(np.float32)
        if not np.isnan(arr).any():
            return arr
        if mode == "zero":
            return np.nan_to_num(arr, nan=0.0)
        if mode != "mean":
            raise ValueError("fill_missing must be 'mean' or 'zero'.")
        out = arr.copy()
        for c in range(out.shape[1]):
            mean = np.nanmean(out[:, c])
            if not np.isfinite(mean):
                mean = 0.0
            out[:, c] = np.nan_to_num(out[:, c], nan=float(mean))
        return out

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_data[idx]),
            torch.from_numpy(self.Y_data[idx]),
            torch.from_numpy(self.mask_data[idx]),
        )


def setup_data_loaders(dataset, batch_size=8, train_fraction=0.8, val_fraction=0.1, shuffle_train=False):
    total_size = len(dataset)
    train_count = int(total_size * train_fraction)
    val_count = max(1, int(total_size * val_fraction))
    train_indices = list(range(0, train_count))
    val_indices = list(range(train_count, min(train_count + val_count, total_size)))
    test_indices = list(range(min(train_count + val_count, total_size), total_size))
    if not test_indices:
        test_indices = val_indices

    train_dl = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle_train)
    val_dl = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    print(f"Samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    return train_dl, val_dl, test_dl


class MaskedPhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        lambda_huber=1.0,
        lambda_temp=0.7,
        lambda_grad=0.3,
        lambda_ssim=0.0,
        huber_delta=0.5,
    ):
        super().__init__()
        self.lambda_huber = lambda_huber
        self.lambda_temp = lambda_temp
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.huber_delta = huber_delta

    @staticmethod
    def masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def huber_loss(self, pred, target, mask):
        err = pred - target
        abs_err = err.abs()
        quadratic = torch.minimum(abs_err, torch.tensor(self.huber_delta, device=pred.device, dtype=pred.dtype))
        linear = abs_err - quadratic
        loss = 0.5 * quadratic.pow(2) + self.huber_delta * linear
        return self.masked_mean(loss, mask)

    def temporal_loss(self, pred, target, mask):
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        diff_mask = mask[:, 1:] * mask[:, :-1]
        return self.masked_mean((diff_pred - diff_target).pow(2), diff_mask)

    def gradient_loss(self, pred, target, mask):
        dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dy_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        dy_mask = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]

        dx_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dx_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        dx_mask = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]

        return self.masked_mean((dx_pred - dx_target).abs(), dx_mask) + self.masked_mean(
            (dy_pred - dy_target).abs(), dy_mask
        )

    def ssim_loss(self, pred, target, mask):
        # Lightweight differentiable proxy: compare local average fields on valid cells.
        b, t, c, h, w = pred.shape
        pred_bt = pred.view(b * t, c, h, w)
        targ_bt = target.view(b * t, c, h, w)
        mask_bt = mask.view(b * t, c, h, w)
        kernel = torch.ones((1, 1, 3, 3), device=pred.device, dtype=pred.dtype) / 9.0
        pred_s = F.conv2d(pred_bt, kernel, padding=1)
        targ_s = F.conv2d(targ_bt, kernel, padding=1)
        return self.masked_mean((pred_s - targ_s).abs(), mask_bt)

    def forward(self, pred, target, mask):
        huber = self.huber_loss(pred, target, mask)
        temporal = self.temporal_loss(pred, target, mask)
        gradient = self.gradient_loss(pred, target, mask)
        ssim_part = self.ssim_loss(pred, target, mask) if self.lambda_ssim else torch.tensor(0.0, device=pred.device)
        total = (
            self.lambda_huber * huber
            + self.lambda_temp * temporal
            + self.lambda_grad * gradient
            + self.lambda_ssim * ssim_part
        )
        return total, huber.detach(), temporal.detach(), gradient.detach()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class DoubleConvSE(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))


class UNetFlexible(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=16, use_se=False):
        super().__init__()
        self.enc1 = DoubleConvSE(in_ch, base, use_se)
        self.enc2 = DoubleConvSE(base, base * 2, use_se)
        self.enc3 = DoubleConvSE(base * 2, base * 4, use_se)
        self.bott = DoubleConvSE(base * 4, base * 8, use_se)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = DoubleConvSE(base * 8, base * 4, use_se)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = DoubleConvSE(base * 4, base * 2, use_se)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = DoubleConvSE(base * 2, base, use_se)
        self.head = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def resize_like(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.resize_like(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self.resize_like(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self.resize_like(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)


class MultiKernelBlock(nn.Module):
    def __init__(self, in_ch, out_per_var=3, kernel_sizes=None):
        super().__init__()
        self.in_ch = int(in_ch)
        self.kernel_sizes = [3, 5, 7] if kernel_sizes is None else [int(k) for k in kernel_sizes]
        if not self.kernel_sizes:
            self.kernel_sizes = [3]
        self.out_per_var = len(self.kernel_sizes)
        self.out_ch = self.in_ch * self.out_per_var
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.in_ch, self.in_ch, k, padding=k // 2, groups=self.in_ch, bias=False)
                for k in self.kernel_sizes
            ]
        )
        self.bn = nn.BatchNorm2d(self.out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat([conv(x) for conv in self.convs], dim=1)))


class CNN_UNet_LSTM(nn.Module):
    def __init__(self, in_ch, height, width, hidden_dim=96, mkb_kernel_sizes=None, use_se=True):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.mkb = MultiKernelBlock(in_ch=in_ch, kernel_sizes=mkb_kernel_sizes)
        self.unet = UNetFlexible(in_ch=self.mkb.out_ch, out_ch=1, base=16, use_se=use_se)
        self.flatten = nn.Flatten(1)
        self.lstm = nn.LSTM(self.height * self.width, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, self.height * self.width)

    def forward(self, x):
        b, t, c, h, w = x.shape
        spatial = []
        for step in range(t):
            feat = self.mkb(x[:, step])
            grid = self.unet(feat)
            spatial.append(self.flatten(grid))
        seq = torch.stack(spatial, dim=1)
        lstm_out, _ = self.lstm(seq)
        y = self.fc(lstm_out.reshape(b * t, -1)).view(b, t, 1, h, w)
        last_spatial = spatial[-1].view(b, 1, h, w)
        return y, last_spatial


def calculate_mae(y_true, y_pred, mask=None):
    yt, yp = masked_numpy(y_true, y_pred, mask)
    return mean_absolute_error(yt, yp)


def calculate_mse(y_true, y_pred, mask=None):
    yt, yp = masked_numpy(y_true, y_pred, mask)
    return mean_squared_error(yt, yp)


def calculate_rmse(y_true, y_pred, mask=None):
    return math.sqrt(calculate_mse(y_true, y_pred, mask))


def calculate_mape(y_true, y_pred, mask=None, epsilon=1e-6):
    yt, yp = masked_numpy(y_true, y_pred, mask)
    return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), epsilon))) * 100.0)


def calculate_ssim(y_true, y_pred, mask=None):
    if ssim is None:
        return float("nan")
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy() if mask is not None else np.ones_like(yt)
    scores = []
    for i in range(yt.shape[0]):
        true_img = np.where(mk[i, 0] > 0, yt[i, 0], np.nan)
        pred_img = np.where(mk[i, 0] > 0, yp[i, 0], np.nan)
        fill_true = np.nanmean(true_img)
        fill_pred = np.nanmean(pred_img)
        true_img = np.nan_to_num(true_img, nan=fill_true)
        pred_img = np.nan_to_num(pred_img, nan=fill_pred)
        data_range = float(np.nanmax(true_img) - np.nanmin(true_img))
        if data_range <= 0:
            data_range = 1.0
        try:
            scores.append(ssim(true_img, pred_img, data_range=data_range, win_size=3))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else float("nan")


def masked_numpy(y_true, y_pred, mask=None):
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    if mask is None:
        return yt.reshape(-1), yp.reshape(-1)
    mk = mask.detach().cpu().numpy() > 0
    return yt[mk], yp[mk]


def evaluate_model(model, data_loader, device, dataset, normalized=False):
    model.eval()
    preds, targets, masks, times = [], [], [], []
    with torch.no_grad():
        for xb, yb, mb in data_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            start = time.time()
            pred, _ = model(xb)
            times.append(time.time() - start)
            preds.append(pred[:, -1].cpu())
            targets.append(yb[:, -1].cpu())
            masks.append(mb[:, -1].cpu())

    pred = torch.cat(preds)
    targ = torch.cat(targets)
    mask = torch.cat(masks)
    if not normalized:
        pred = unnormalize_target(pred, dataset)
        targ = unnormalize_target(targ, dataset)

    return {
        "MAE": calculate_mae(targ, pred, mask),
        "MSE": calculate_mse(targ, pred, mask),
        "RMSE": calculate_rmse(targ, pred, mask),
        "MAPE": calculate_mape(targ, pred, mask),
        "SSIM": calculate_ssim(targ, pred, mask),
        "Avg_Inference_Time(s)": float(np.mean(times)) if times else float("nan"),
    }


def unnormalize_target(tensor, dataset):
    mu = torch.tensor(dataset.y_mu.reshape(1, 1, 1), dtype=tensor.dtype, device=tensor.device)
    sd = torch.tensor(dataset.y_sd.reshape(1, 1, 1), dtype=tensor.dtype, device=tensor.device)
    return (tensor * sd) + mu


def metrics_per_horizon(model, data_loader, device, dataset):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for xb, yb, mb in data_loader:
            pred, _ = model(xb.to(device))
            preds.append(pred.cpu())
            targets.append(yb)
            masks.append(mb)
    pred = unnormalize_target(torch.cat(preds), dataset)
    targ = unnormalize_target(torch.cat(targets), dataset)
    mask = torch.cat(masks)
    out = {"MSE": [], "RMSE": [], "SSIM": []}
    for h in range(pred.shape[1]):
        out["MSE"].append(calculate_mse(targ[:, h], pred[:, h], mask[:, h]))
        out["RMSE"].append(calculate_rmse(targ[:, h], pred[:, h], mask[:, h]))
        out["SSIM"].append(calculate_ssim(targ[:, h], pred[:, h], mask[:, h]))
    return out


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    num_epochs=100,
    lr=5e-4,
    patience=15,
    lambda_huber=1.0,
    lambda_temp=0.7,
    lambda_grad=0.3,
    lambda_ssim=0.1,
    huber_delta=0.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = MaskedPhysicsInformedLoss(lambda_huber, lambda_temp, lambda_grad, lambda_ssim, huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_huber": [],
        "train_temp": [],
        "train_grad": [],
        "test_eval_epochs": [],
        "test_mae": [],
        "test_mse": [],
        "test_rmse": [],
        "test_ssim": [],
    }
    best_state = None
    best_val = float("inf")
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        totals = []
        components = []
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss, huber, temp, grad = criterion(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            components.append([float(huber.cpu()), float(temp.cpu()), float(grad.cpu())])

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred, _ = model(xb)
                loss, _, _, _ = criterion(pred, yb, mb)
                val_losses.append(float(loss.cpu()))

        scheduler.step()
        train_loss = float(np.mean(totals))
        val_loss = float(np.mean(val_losses))
        comp = np.mean(np.asarray(components), axis=0)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_huber"].append(float(comp[0]))
        history["train_temp"].append(float(comp[1]))
        history["train_grad"].append(float(comp[2]))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_rainfall_mkcnn_unet_lstm.pth")
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{num_epochs} train={train_loss:.4f} val={val_loss:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_trained"] = len(history["train_losses"])
    return model, history


def plot_training_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_losses"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_losses"], label="Train", linewidth=1.8)
    axes[0].plot(epochs, history["val_losses"], label="Validation", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, history["train_huber"], label="Magnitude")
    axes[1].plot(epochs, history["train_temp"], label="Temporal")
    axes[1].plot(epochs, history["train_grad"], label="Gradient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component Loss")
    axes[1].set_title("Physics Loss Components")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_loss.png", dpi=300)
    plt.close(fig)


def plot_metrics_vs_horizon(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, color in [("MSE", "tab:blue"), ("RMSE", "tab:green"), ("SSIM", "tab:orange")]:
        vals = metrics.get(key, [])
        if not vals:
            continue
        xs = np.arange(1, len(vals) + 1)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(xs, vals, marker="o", color=color, linewidth=1.8)
        ax.set_xlabel("Horizon")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs Prediction Horizon")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key.lower()}_vs_horizon.png", dpi=300)
        plt.close(fig)


def plot_prediction_maps(model, data_loader, device, dataset, output_dir, max_samples=3):
    output_dir = Path(output_dir) / "prediction_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    lons = dataset.lon_values
    lats = dataset.lat_values
    count = 0
    with torch.no_grad():
        for xb, yb, mb in data_loader:
            pred, spatial = model(xb.to(device))
            pred = unnormalize_target(pred[:, -1].cpu(), dataset)
            targ = unnormalize_target(yb[:, -1], dataset)
            mask = mb[:, -1]
            for i in range(pred.shape[0]):
                if count >= max_samples:
                    return
                p = np.where(mask[i, 0].numpy() > 0, pred[i, 0].numpy(), np.nan)
                t = np.where(mask[i, 0].numpy() > 0, targ[i, 0].numpy(), np.nan)
                err = p - t
                vmin = np.nanmin([np.nanmin(p), np.nanmin(t)])
                vmax = np.nanmax([np.nanmax(p), np.nanmax(t)])
                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                for ax, data, title, cmap in [
                    (axes[0], t, "Actual Rainfall", "Blues"),
                    (axes[1], p, "Predicted Rainfall", "Blues"),
                    (axes[2], err, "Prediction Error", "RdBu_r"),
                ]:
                    im = ax.imshow(data, origin="upper", extent=[min(lons), max(lons), min(lats), max(lats)], cmap=cmap)
                    if title != "Prediction Error":
                        im.set_clim(vmin, vmax)
                    ax.set_title(title)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    fig.colorbar(im, ax=ax, fraction=0.046)
                plt.tight_layout()
                plt.savefig(output_dir / f"rainfall_prediction_sample_{count + 1}.png", dpi=300)
                plt.close(fig)
                count += 1


def plot_mkb_features(model, data_loader, device, dataset, output_dir, max_samples=2):
    output_dir = Path(output_dir) / "mkb_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        count = 0
        for xb, _, mb in data_loader:
            xb = xb.to(device)
            feat = model.mkb(xb[:, -1])
            for sample_idx in range(feat.shape[0]):
                if count >= max_samples:
                    return
                n_show = min(12, feat.shape[1])
                cols = 4
                rows = int(math.ceil(n_show / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
                axes = np.asarray(axes).reshape(-1)
                mask = mb[sample_idx, -1, 0].numpy() > 0
                for k in range(n_show):
                    data = feat[sample_idx, k].cpu().numpy()
                    data = np.where(mask, data, np.nan)
                    im = axes[k].imshow(data, origin="upper", cmap="RdBu_r")
                    axes[k].set_title(f"MKB channel {k}")
                    axes[k].axis("off")
                    fig.colorbar(im, ax=axes[k], fraction=0.046)
                for k in range(n_show, len(axes)):
                    axes[k].axis("off")
                plt.tight_layout()
                plt.savefig(output_dir / f"mkb_features_sample_{count + 1}.png", dpi=300)
                plt.close(fig)
                count += 1


def run_ablation(args, input_specs):
    output_root = Path(args.output_dir) / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for keep_name, _ in input_specs:
        specs = [item for item in input_specs if item[0] == keep_name]
        print(f"\nAblation run: {keep_name}")
        run_dir = output_root / keep_name
        results = run_pipeline(args, specs, run_dir, make_plots=False)
        rows.append({"input": keep_name, **results})
    all_results = run_pipeline(args, input_specs, output_root / "all_inputs", make_plots=False)
    rows.append({"input": "all_inputs", **all_results})
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "ablation_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["input"], df["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Rainfall Input Ablation")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_root / "ablation_rmse.png", dpi=300)
    plt.close(fig)
    return df


def run_pipeline(args, input_specs, output_dir=None, make_plots=True):
    output_dir = Path(output_dir or args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RainfallGridDataset(
        input_specs=input_specs,
        target_path=args.target,
        sequence_length=args.sequence_length,
        target_delay=args.target_delay,
        include_month_channels=not args.no_month_channels,
        fill_missing=args.fill_missing,
        predict_delta=args.predict_delta,
    )
    train_dl, val_dl, test_dl = setup_data_loaders(
        dataset,
        batch_size=args.batch_size,
        shuffle_train=args.shuffle_train,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel_sizes = [int(x) for x in args.mkb_kernel_sizes.split(",") if x.strip()]
    model = CNN_UNet_LSTM(
        in_ch=dataset.in_ch,
        height=dataset.height,
        width=dataset.width,
        hidden_dim=args.hidden_dim,
        mkb_kernel_sizes=kernel_sizes,
        use_se=args.use_se,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    model, history = train_model(
        model,
        train_dl,
        val_dl,
        device,
        output_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=args.patience,
        lambda_huber=args.lambda_huber,
        lambda_temp=args.lambda_temp,
        lambda_grad=args.lambda_grad,
        lambda_ssim=args.lambda_ssim,
        huber_delta=args.huber_delta,
    )
    results = evaluate_model(model, test_dl, device, dataset, normalized=False)
    results_norm = evaluate_model(model, test_dl, device, dataset, normalized=True)
    pd.DataFrame([results]).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame([results_norm]).to_csv(output_dir / "test_metrics_normalized.csv", index=False)
    print("\nTest metrics:")
    for key, val in results.items():
        print(f"{key}: {val:.4f}")

    if make_plots:
        plot_training_history(history, output_dir)
        horizon_metrics = metrics_per_horizon(model, test_dl, device, dataset)
        plot_metrics_vs_horizon(horizon_metrics, output_dir)
        plot_prediction_maps(model, test_dl, device, dataset, output_dir, max_samples=args.plot_samples)
        plot_mkb_features(model, test_dl, device, dataset, output_dir, max_samples=args.plot_samples)

    return results


def parse_input_specs(text):
    if not text:
        return DEFAULT_INPUTS
    specs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError("Each input must be name=/path/to/file.csv")
        name, path = chunk.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Rainfall MKCNN-U-Net-LSTM training pipeline")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--inputs",
        type=str,
        default="",
        help="Comma-separated list like v10=file.csv,u10=file.csv,t2m=file.csv. Defaults to Somya inputs.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("rainfall_output_plots"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sequence_length", type=int, default=6)
    parser.add_argument("--target_delay", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_huber", type=float, default=1.0)
    parser.add_argument("--lambda_temp", type=float, default=0.7)
    parser.add_argument("--lambda_grad", type=float, default=0.3)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--mkb_kernel_sizes", type=str, default="3,5,7")
    parser.add_argument("--use_se", action="store_true", default=True)
    parser.add_argument("--no_month_channels", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--predict_delta", action="store_true")
    parser.add_argument("--fill_missing", choices=["mean", "zero"], default="mean")
    parser.add_argument("--plot_samples", type=int, default=3)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    input_specs = parse_input_specs(args.inputs)
    if args.ablation:
        run_ablation(args, input_specs)
    else:
        run_pipeline(args, input_specs)


if __name__ == "__main__":
    main()
