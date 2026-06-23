"""
Training loop + experiment harness.

Each call to train_one_experiment() trains a fresh model with a given
config and returns per-epoch history (train loss, val loss, val accuracy).

run_all_experiments() runs a predefined list of configs (baseline +
comparisons), saves a results table (CSV) and comparison plots (PNG)
into results/.
"""

import os
import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import SmallCNN
from dataset_utils import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "sgd_no_momentum":
        return torch.optim.SGD(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_one_experiment(config, dataset_dir="dataset", epochs=15, verbose=True):
    """
    config keys (all optional, with defaults):
        name            - str, label for this run
        lr              - float, learning rate (default 0.001)
        batch_size      - int (default 16)
        dropout         - float (default 0.5)
        optimizer       - "adam" | "sgd" | "sgd_no_momentum" (default "adam")
        weight_decay    - float, L2 regularization strength (default 0.0)
        augment         - bool, data augmentation on/off (default False)

    Returns a dict with the config plus per-epoch history lists.
    """
    name = config.get("name", "unnamed")
    lr = config.get("lr", 0.001)
    batch_size = config.get("batch_size", 16)
    dropout = config.get("dropout", 0.5)
    optimizer_name = config.get("optimizer", "adam")
    weight_decay = config.get("weight_decay", 0.0)
    augment = config.get("augment", False)

    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"  lr={lr} batch_size={batch_size} dropout={dropout} "
          f"optimizer={optimizer_name} weight_decay={weight_decay} augment={augment}")
    print(f"{'='*60}")

    train_loader, val_loader, class_names = get_dataloaders(
        dataset_dir=dataset_dir, batch_size=batch_size, augment=augment
    )

    model = SmallCNN(num_classes=len(class_names), dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "sgd_no_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []
    }

    start_time = time.time()

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validation ----
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Flag instability: loss exploding or becoming NaN
        unstable_flag = ""
        if train_loss != train_loss:  # NaN check
            unstable_flag = "  <-- NaN! Training has diverged."
        elif epoch > 0 and train_loss > history["train_loss"][epoch - 1] * 3:
            unstable_flag = "  <-- Loss spiked sharply."

        if verbose:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{unstable_flag}")

        # stop early if things have completely diverged (saves time, avoids junk plots)
        if train_loss != train_loss:
            print("Stopping early: loss became NaN.")
            break

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")

    result = {
        "config": config,
        "history": history,
        "class_names": class_names,
        "elapsed_seconds": elapsed,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
    }
    return result


def plot_comparison(results, key_a, key_b, metric, title, filename):
    """
    Plots one metric (e.g. 'val_loss') for two experiments side by side
    on the same axes, for direct visual comparison.
    """
    plt.figure(figsize=(8, 5))
    for key in (key_a, key_b):
        r = results[key]
        epochs_range = range(1, len(r["history"][metric]) + 1)
        plt.plot(epochs_range, r["history"][metric], marker="o", label=r["config"]["name"])

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def save_summary_table(results, filename="summary.csv"):
    import csv
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "lr", "batch_size", "dropout", "optimizer",
                          "weight_decay", "augment", "final_val_acc", "final_train_loss",
                          "final_val_loss", "epochs_run", "seconds"])
        for key, r in results.items():
            c = r["config"]
            writer.writerow([
                c.get("name"), c.get("lr", 0.001), c.get("batch_size", 16),
                c.get("dropout", 0.5), c.get("optimizer", "adam"),
                c.get("weight_decay", 0.0), c.get("augment", False),
                round(r["final_val_acc"], 4) if r["final_val_acc"] is not None else "N/A",
                round(r["history"]["train_loss"][-1], 4) if r["history"]["train_loss"] else "N/A",
                round(r["history"]["val_loss"][-1], 4) if r["history"]["val_loss"] else "N/A",
                len(r["history"]["train_loss"]),
                round(r["elapsed_seconds"], 1),
            ])
    print(f"Saved summary table: {path}")


def run_all_experiments(dataset_dir="dataset", epochs=15):
    """
    Runs baseline + comparison experiments. Edit this list to add/remove
    experiments as you explore.
    """
    configs = {
        "baseline":        {"name": "Baseline (Adam, lr=0.001, bs=16, dropout=0.5)"},

        # --- Learning rate comparison ---
        "lr_low":          {"name": "LR too low (0.00005)", "lr": 0.00005},
        "lr_high_unstable":{"name": "LR too high (1.0) -- UNSTABLE", "lr": 1.0},

        # --- Batch size comparison ---
        "batch_small":     {"name": "Small batch (4)", "batch_size": 4},
        "batch_large":     {"name": "Large batch (32)", "batch_size": 32},

        # --- Dropout comparison ---
        "no_dropout":      {"name": "No dropout (0.0)", "dropout": 0.0},
        "high_dropout":    {"name": "High dropout (0.7)", "dropout": 0.7},

        # --- Optimizer comparison ---
        "sgd_plain":       {"name": "SGD, no momentum", "optimizer": "sgd_no_momentum"},

        # --- Regularization (weight decay) ---
        "weight_decay":    {"name": "With weight decay (1e-3)", "weight_decay": 1e-3},

        # --- Data augmentation ---
        "augmented":       {"name": "With data augmentation", "augment": True},
    }

    results = {}
    for key, cfg in configs.items():
        results[key] = train_one_experiment(cfg, dataset_dir=dataset_dir, epochs=epochs)

    # Save raw results as JSON (for the README / further analysis)
    json_safe = {
        k: {
            "config": v["config"],
            "history": v["history"],
            "final_val_acc": v["final_val_acc"],
            "elapsed_seconds": v["elapsed_seconds"],
        }
        for k, v in results.items()
    }
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(json_safe, f, indent=2)

    save_summary_table(results)

    # Key comparison plots
    plot_comparison(results, "baseline", "lr_low", "train_loss",
                     "Learning Rate Too Low: Train Loss", "lr_low_vs_baseline_trainloss.png")
    plot_comparison(results, "baseline", "lr_high_unstable", "train_loss",
                     "Learning Rate Too High (Unstable): Train Loss", "lr_high_vs_baseline_trainloss.png")
    plot_comparison(results, "batch_small", "batch_large", "val_loss",
                     "Batch Size Effect: Validation Loss", "batch_size_val_loss.png")
    plot_comparison(results, "no_dropout", "high_dropout", "val_loss",
                     "Dropout Effect: Validation Loss", "dropout_val_loss.png")
    plot_comparison(results, "no_dropout", "high_dropout", "train_loss",
                     "Dropout Effect: Train Loss (overfitting check)", "dropout_train_loss.png")
    plot_comparison(results, "baseline", "sgd_plain", "val_acc",
                     "Optimizer Effect: Validation Accuracy", "optimizer_val_acc.png")
    plot_comparison(results, "baseline", "weight_decay", "val_loss",
                     "Weight Decay Effect: Validation Loss", "weight_decay_val_loss.png")
    plot_comparison(results, "baseline", "augmented", "val_acc",
                     "Data Augmentation Effect: Validation Accuracy", "augmentation_val_acc.png")

    print("\nAll experiments complete. Check the results/ folder for plots, summary.csv, and all_results.json")
    return results


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    run_all_experiments(epochs=15)
