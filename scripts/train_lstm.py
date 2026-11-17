"""
Week 2 deliverable: per-appliance LSTM training + evaluation.

Architecture (per Ch3.8 methodology, agreed scope):
    Input:  96 timesteps (24h of 15-min buckets), 1 feature (scaled power)
    Layer1: LSTM(64, return_sequences=True), dropout 0.2
    Layer2: LSTM(64), dropout 0.2
    Dense:  4 units (next 1h = 4 x 15-min buckets)
    Loss:   MSE, Optimizer: Adam, EarlyStopping on val_loss (patience=8)

Usage:
    python train_lstm.py --appliance fan
    python train_lstm.py --all
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Must match preprocess.py -- revised Week 3 from 96/4 (24h/1h) to 8/2 (2h/30min)
# after real iAWE data showed session-based gaps that made 24h windows infeasible.
INPUT_STEPS = 8
OUTPUT_STEPS = 2
N_FEATURES = 1  # power only -- tested 5 (with cyclical time feats) in Week 3, univariate won on 2/3 appliances, see preprocess.py note
LSTM_UNITS = 64
DROPOUT = 0.2
EPOCHS = 100  # production default; override via --epochs for constrained dev runs
BATCH_SIZE = 32
# Bumped from 8 -> 20 (Week 3): with 5 input features now (vs 1 originally)
# and small sample counts, patience=8 was stopping training prematurely
# (computer stopped at epoch 9, badly underfit, R2=0.05). patience=20 let it
# run to epoch 42 and recovered R2=0.38 -- confirmed this wasn't noise before
# keeping it as the new default.
PATIENCE = 20


def load_splits(appliance: str):
    d = PROCESSED_DIR / appliance
    X_train = np.load(d / "X_train.npy")
    y_train = np.load(d / "y_train.npy")
    X_val = np.load(d / "X_val.npy")
    y_val = np.load(d / "y_val.npy")
    X_test = np.load(d / "X_test.npy")
    y_test = np.load(d / "y_test.npy")
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # X arrays are already (samples, timesteps, 5) -- [power, hour_sin,
    # hour_cos, dow_sin, dow_cos] per timestep, built in preprocess.py.
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(INPUT_STEPS, N_FEATURES)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dropout(DROPOUT),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(OUTPUT_STEPS),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(y_true_scaled, y_pred_scaled, scaler):
    """Inverse-transform to real Watts before computing metrics -- metrics
    reported in the thesis must be in physical units, not 0-1 scaled space."""
    n_out = y_true_scaled.shape[1]
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1, n_out)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, n_out)

    rmse = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
    mae = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    r2 = float(r2_score(y_true.ravel(), y_pred.ravel()))

    # NRMSE (normalized by mean of |actual|) -- robust when the test slice
    # happens to have near-zero variance, which makes R2 blow up to a huge
    # meaningless number even when the absolute error is small. Real risk on
    # session-based real-world data where a chronological split can land a
    # near-idle stretch in the test set (this happened on the geyser/AC
    # channel in this project -- flag it, don't hide it).
    mean_abs_actual = float(np.mean(np.abs(y_true)))
    nrmse_pct = float(rmse / mean_abs_actual * 100) if mean_abs_actual > 1e-6 else float("nan")
    test_std = float(np.std(y_true))
    low_variance_warning = test_std < 0.01 * (float(np.max(y_true)) - float(np.min(y_true)) + 1e-9) if (np.max(y_true) - np.min(y_true)) > 1 else test_std < 1.0

    metrics = {
        "rmse_watts": rmse,
        "mae_watts": mae,
        "r2": r2,
        "nrmse_pct": nrmse_pct,
        "test_std_watts": test_std,
        "r2_reliable": not low_variance_warning,
    }
    return metrics, y_true, y_pred


def plot_history(history, appliance):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title(f"LSTM Training Loss — {appliance}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{appliance}_loss_curve.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_predictions(y_true, y_pred, appliance, n_show=200):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.ravel()[:n_show], label="Actual", linewidth=1)
    plt.plot(y_pred.ravel()[:n_show], label="Predicted", linewidth=1, alpha=0.8)
    plt.title(f"LSTM Prediction vs Actual — {appliance} (test set, first {n_show} pts)")
    plt.xlabel("15-min step")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{appliance}_prediction_vs_actual.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def train_appliance(appliance: str, epochs: int = EPOCHS, patience: int = PATIENCE):
    print(f"\n=== Training LSTM: {appliance} ===")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_splits(appliance)
    print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    model = build_model()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    n_epochs_run = len(history.history["loss"])
    print(f"  stopped after {n_epochs_run} epochs "
          f"(best val_loss={min(history.history['val_loss']):.5f})")

    y_pred_scaled = model.predict(X_test, verbose=0)
    metrics, y_true_w, y_pred_w = evaluate(y_test, y_pred_scaled, scaler)
    r2_flag = "" if metrics["r2_reliable"] else "  [R2 UNRELIABLE -- near-zero test variance, use NRMSE instead]"
    print(f"  TEST metrics (Watts): RMSE={metrics['rmse_watts']:.2f} "
          f"MAE={metrics['mae_watts']:.2f} R2={metrics['r2']:.4f} "
          f"NRMSE={metrics['nrmse_pct']:.1f}%{r2_flag}")

    model_path = MODELS_DIR / f"lstm_{appliance}.keras"
    model.save(model_path)

    loss_plot = plot_history(history, appliance)
    pred_plot = plot_predictions(y_true_w, y_pred_w, appliance)

    result = {
        "appliance": appliance,
        "architecture": f"LSTM({LSTM_UNITS})x2 + Dropout({DROPOUT}) + Dense({OUTPUT_STEPS})",
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "epochs_run": n_epochs_run,
        "epochs_max": EPOCHS,
        "early_stop_patience": PATIENCE,
        "metrics": metrics,
        "model_path": str(model_path),
        "loss_curve": str(loss_plot),
        "prediction_plot": str(pred_plot),
    }

    with open(RESULTS_DIR / f"{appliance}_lstm_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  saved model -> {model_path.name}, results -> {appliance}_lstm_results.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", choices=["fan", "geyser", "computer"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    appliances = ["fan", "geyser", "computer"] if (args.all or not args.appliance) else [args.appliance]
    all_results = [train_appliance(a, epochs=args.epochs, patience=args.patience) for a in appliances]

    print("\n=== Summary ===")
    for r in all_results:
        m = r["metrics"]
        flag = "" if m["r2_reliable"] else " [R2 unreliable]"
        print(f"  {r['appliance']:10s} RMSE={m['rmse_watts']:8.2f}W  "
              f"MAE={m['mae_watts']:8.2f}W  R2={m['r2']:.4f}  NRMSE={m['nrmse_pct']:.1f}%  "
              f"(epochs={r['epochs_run']}){flag}")


if __name__ == "__main__":
    main()
