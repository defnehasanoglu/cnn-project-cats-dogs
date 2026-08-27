"""
Cats vs Dogs CNN Classifier
Transfer Learning with EfficientNetB0

HPC-ready: verbose checkpointing at every stage so you can pinpoint
exactly where a job failed from the SLURM / PBS log file.
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC nodes
import matplotlib.pyplot as plt

# Disable Kitty terminal capability checks by forcing a standard terminal type
os.environ["TERM"] = "dumb"

# Alternatively, mock psutil's process navigation if TERM doesn't bypass it completely
import psutil
def mock_parent():
    return None
psutil.Process.parent = staticmethod(mock_parent)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOGGER
# Writes timestamped entries to both stdout AND checkpoints/run.log
# so you can tail -f the log file on the cluster while the job runs.
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/epoch_weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("checkpoints/run.log", mode="a"),
    ],
)
log = logging.getLogger("train")

_STAGE_FILE = "checkpoints/completed_stages.json"

def _load_stages() -> dict:
    if os.path.exists(_STAGE_FILE):
        with open(_STAGE_FILE) as f:
            return json.load(f)
    return {}

def _save_stages(stages: dict):
    with open(_STAGE_FILE, "w") as f:
        json.dump(stages, f, indent=2)

def checkpoint(stage: str, detail: str = ""):
    """Mark a stage as completed and log it."""
    stages = _load_stages()
    stages[stage] = {"done": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "detail": detail}
    _save_stages(stages)
    msg = f"CHECKPOINT [{stage}]"
    if detail:
        msg += f" — {detail}"
    log.info(msg)

def is_done(stage: str) -> bool:
    """Return True if this stage was already completed (useful for resuming)."""
    return _load_stages().get(stage, {}).get("done", False)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "data_dir":         "PetImages",
    "img_size":         (224, 224),
    "batch_size":       32,
    "epochs":           20,
    "fine_tune_epochs": 10,
    "learning_rate":    1e-3,
    "fine_tune_lr":     1e-5,
    "dropout":          0.4,
    "val_split":        0.15,
    "test_split":       0.15,
    "num_classes":      2,
    "class_names":      ["Cat", "Dog"],
    "model_save_path":  "checkpoints/best_model.keras",
    "phase1_ckpt_dir":  "checkpoints/epoch_weights/phase1",
    "phase2_ckpt_dir":  "checkpoints/epoch_weights/phase2",
}

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGS — lets the SAME script run as a tiny local smoke test
# or the full HPC job, no code duplication.
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Cats vs Dogs training")
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run a tiny local sanity check: small dataset, 1-2 epochs, "
             "no fine-tuning. Use this before submitting to HPC."
    )
    p.add_argument(
        "--data-dir", type=str, default=None,
        help="Override the dataset directory (default: PetImages, "
             "or PetImages_mini when --smoke-test is set)."
    )
    return p.parse_args()


def apply_smoke_test_overrides(config: dict, user_data_dir: str = None):
    """Shrink the run drastically so it finishes in ~1 minute on a laptop."""
    config["data_dir"]         = user_data_dir or "PetImages_mini"
    config["epochs"]           = 2
    config["fine_tune_epochs"] = 1
    config["val_split"]        = 0.25
    config["test_split"]       = 0.25
    config["batch_size"]       = 8
    log.info("⚠ SMOKE TEST MODE — using tiny dataset & epoch counts")
    log.info("  This only checks the code runs end-to-end, NOT model quality.")
    return config


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Environment & GPU
# ══════════════════════════════════════════════════════════════════════════════

def stage_environment():
    log.info("━" * 60)
    log.info("STAGE 1 — Environment check")
    log.info("━" * 60)

    log.info(f"Python       : {sys.version}")
    log.info(f"TensorFlow   : {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"GPUs visible : {len(gpus)}")
    for g in gpus:
        log.info(f"  {g}")

    if not gpus:
        log.warning("No GPU found — training will run on CPU (slow on HPC!)")
    else:
        # Allow memory growth so multiple jobs can share a node
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                log.info(f"  Memory growth enabled on {gpu.name}")
            except RuntimeError as e:
                log.warning(f"  Could not set memory growth: {e}")

    log.info(f"Working dir  : {os.getcwd()}")
    log.info(f"Data dir     : {os.path.abspath(CONFIG['data_dir'])}")

    checkpoint("env_check", f"TF {tf.__version__}, {len(gpus)} GPU(s)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Data loading & pipeline
# ══════════════════════════════════════════════════════════════════════════════

def stage_data() -> tuple:
    log.info("━" * 60)
    log.info("STAGE 2 — Data loading & pipeline")
    log.info("━" * 60)

    data_dir = CONFIG["data_dir"]
    if not os.path.isdir(data_dir):
        log.error(f"Data directory not found: {os.path.abspath(data_dir)}")
        log.error("Run prepare_dataset.py first, or check your HPC data path.")
        sys.exit(1)

    log.info(f"Reading images from: {data_dir}")

    t0 = time.time()
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        color_mode="rgb",
        image_size=CONFIG["img_size"],
        batch_size=None,
        shuffle=True,
        seed=SEED,
    )
    log.info(f"Directory scan done in {time.time()-t0:.1f}s")

    dataset_size = full_ds.cardinality().numpy()
    val_size   = int(dataset_size * CONFIG["val_split"])
    test_size  = int(dataset_size * CONFIG["test_split"])
    train_size = dataset_size - val_size - test_size

    log.info(f"Total images : {dataset_size}")
    log.info(f"Train        : {train_size}")
    log.info(f"Val          : {val_size}")
    log.info(f"Test         : {test_size}")

    train_raw = full_ds.take(train_size)
    val_raw   = full_ds.skip(train_size).take(val_size)
    test_raw  = full_ds.skip(train_size + val_size)

    # ── Augmentation ──────────────────────────────────────────────────────────
    log.info("Building augmentation pipeline...")
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    AUTOTUNE = tf.data.AUTOTUNE

    def apply_augment(x, y):
        return augment(x, training=True), y

    train_ds = (train_raw.map(apply_augment, num_parallel_calls=AUTOTUNE)
                         .batch(CONFIG["batch_size"]).prefetch(AUTOTUNE))
    val_ds   = val_raw.batch(CONFIG["batch_size"]).prefetch(AUTOTUNE)
    test_ds  = test_raw.batch(CONFIG["batch_size"]).prefetch(AUTOTUNE)

    log.info(f"Train batches : {train_ds.cardinality().numpy()}")
    log.info(f"Val   batches : {val_ds.cardinality().numpy()}")
    log.info(f"Test  batches : {test_ds.cardinality().numpy()}")

    # Sanity-check one batch so we catch shape/dtype errors before training
    log.info("Pulling one batch to verify pipeline...")
    sample_x, sample_y = next(iter(train_ds))
    log.info(f"  Batch shape  : images={sample_x.shape}  labels={sample_y.shape}")
    log.info(f"  Pixel range  : [{sample_x.numpy().min():.1f}, {sample_x.numpy().max():.1f}]")
    log.info(f"  Label dtype  : {sample_y.dtype}")

    checkpoint("data_pipeline",
               f"{train_size} train / {val_size} val / {test_size} test images")
    return train_ds, val_ds, test_ds


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Model construction
# ══════════════════════════════════════════════════════════════════════════════

def stage_build_model() -> tf.keras.Model:
    log.info("━" * 60)
    log.info("STAGE 3 — Model construction (EfficientNetB0 backbone)")
    log.info("━" * 60)

    log.info("Loading EfficientNetB0 with ImageNet weights...")
    t0 = time.time()

    inputs  = layers.Input(shape=(*CONFIG["img_size"], 3), name="input_image")
    backbone = EfficientNetB0(include_top=False, weights="imagenet",
                              input_tensor=inputs)
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout"], name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="CatsVsDogs_EfficientNetB0")

    log.info(f"Backbone loaded in {time.time()-t0:.1f}s")
    log.info(f"Total params     : {model.count_params():,}")
    log.info(f"Trainable params : {sum(np.prod(v.shape) for v in model.trainable_variables):,}")
    log.info(f"Frozen params    : {sum(np.prod(v.shape) for v in model.non_trainable_variables):,}")

    # Save model architecture diagram
    try:
        tf.keras.utils.plot_model(model, to_file="checkpoints/model_architecture.png",
                                  show_shapes=True, show_layer_names=True)
        log.info("Model diagram saved to checkpoints/model_architecture.png")
    except Exception as e:
        log.warning(f"Could not save model diagram (pydot/graphviz missing?): {e}")

    checkpoint("model_built", f"{model.count_params():,} total params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# COMPILE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def compile_model(model: tf.keras.Model, lr: float, phase: str) -> tf.keras.Model:
    log.info(f"Compiling model — lr={lr:.2e}  phase={phase}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    checkpoint(f"compile_{phase}", f"lr={lr:.2e}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — per-epoch weight saving + full suite
# ══════════════════════════════════════════════════════════════════════════════

class HPCProgressLogger(callbacks.Callback):
    """Logs metrics after every epoch to the checkpoint log file."""

    def on_epoch_begin(self, epoch, logs=None):
        log.info(f"  ── Epoch {epoch + 1} starting...")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        parts = [f"epoch={epoch+1}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.4f}")
        log.info("  " + "  ".join(parts))


def get_callbacks(phase: str) -> list:
    ckpt_dir = CONFIG[f"{phase}_ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    return [
        # ── Best model (by val AUC) ───────────────────────────────────────────
        callbacks.ModelCheckpoint(
            filepath=CONFIG["model_save_path"],
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # ── Every-epoch weight snapshot (for crash recovery) ──────────────────
        callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "epoch_{epoch:03d}_valauc{val_auc:.4f}.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=False,       # save EVERY epoch
            save_weights_only=True,     # smaller files on HPC scratch
            verbose=0,
        ),
        # ── Early stopping ────────────────────────────────────────────────────
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # ── LR reduction ──────────────────────────────────────────────────────
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # ── CSV log (parseable metrics file for post-job analysis) ────────────
        callbacks.CSVLogger(
            filename=f"checkpoints/{phase}_metrics.csv",
            append=True,
        ),
        # ── TensorBoard ───────────────────────────────────────────────────────
        callbacks.TensorBoard(
            log_dir=f"logs/{phase}",
            histogram_freq=1,
            profile_batch=0,            # disable profiler (can hang on some HPC)
        ),
        # ── HPC epoch logger ──────────────────────────────────────────────────
        HPCProgressLogger(),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Phase 1: Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def stage_phase1(model, train_ds, val_ds):
    log.info("━" * 60)
    log.info("STAGE 4 — Phase 1: Feature Extraction (backbone frozen)")
    log.info("━" * 60)
    log.info(f"Epochs planned : {CONFIG['epochs']}")
    log.info(f"Learning rate  : {CONFIG['learning_rate']:.2e}")
    log.info(f"Per-epoch weights → {CONFIG['phase1_ckpt_dir']}/")

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )
    elapsed = time.time() - t0

    best_auc = max(history.history.get("val_auc", [0]))
    best_acc = max(history.history.get("val_accuracy", [0]))
    epochs_ran = len(history.history["loss"])

    log.info(f"Phase 1 done in {elapsed/60:.1f} min  ({epochs_ran} epochs)")
    log.info(f"Best val_auc      : {best_auc:.4f}")
    log.info(f"Best val_accuracy : {best_acc:.4f}")

    checkpoint("phase1_training",
               f"{epochs_ran} epochs, best val_auc={best_auc:.4f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Fine-tuning setup
# ══════════════════════════════════════════════════════════════════════════════

def stage_enable_fine_tuning(model, unfreeze_from: int = 100):
    log.info("━" * 60)
    log.info("STAGE 5 — Enabling fine-tuning (unfreezing top backbone layers)")
    log.info("━" * 60)

    # Step 1: Scan all layers in the model
    # Find backbone layers, excluding the output (head) layers
    head_layer_names = ['gap', 'batch_normalization', 'dropout', 'output']
    
    # All layers whose names are NOT in head_layer_names are our backbone layers
    backbone_layers = [layer for layer in model.layers if layer.name not in head_layer_names]
    
    # Step 2: First, make all backbone layers trainable
    for layer in backbone_layers:
        layer.trainable = True

    # Step 3: Freeze the initial layers back up to the specified limit (unfreeze_from)
    for layer in backbone_layers[:unfreeze_from]:
        layer.trainable = False

    total_backbone = len(backbone_layers)
    unfrozen = total_backbone - unfreeze_from
    log.info(f"Backbone layers   : {total_backbone}")
    log.info(f"Frozen up to      : layer {unfreeze_from}")
    log.info(f"Trainable layers : {unfrozen} backbone + head")

    checkpoint("fine_tune_enabled",
               f"unfrozen {unfrozen}/{total_backbone} backbone layers")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Phase 2: Fine-Tuning
# ══════════════════════════════════════════════════════════════════════════════

def stage_phase2(model, train_ds, val_ds):
    log.info("━" * 60)
    log.info("STAGE 6 — Phase 2: Fine-Tuning")
    log.info("━" * 60)
    log.info(f"Epochs planned : {CONFIG['fine_tune_epochs']}")
    log.info(f"Learning rate  : {CONFIG['fine_tune_lr']:.2e}")
    log.info(f"Per-epoch weights → {CONFIG['phase2_ckpt_dir']}/")

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["fine_tune_epochs"],
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )
    elapsed = time.time() - t0

    best_auc = max(history.history.get("val_auc", [0]))
    best_acc = max(history.history.get("val_accuracy", [0]))
    epochs_ran = len(history.history["loss"])

    log.info(f"Phase 2 done in {elapsed/60:.1f} min  ({epochs_ran} epochs)")
    log.info(f"Best val_auc      : {best_auc:.4f}")
    log.info(f"Best val_accuracy : {best_acc:.4f}")

    checkpoint("phase2_training",
               f"{epochs_ran} epochs, best val_auc={best_auc:.4f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Plotting
# ══════════════════════════════════════════════════════════════════════════════

def stage_plot(history_p1, history_p2):
    log.info("━" * 60)
    log.info("STAGE 7 — Training curves")
    log.info("━" * 60)

    combined = {k: list(v) for k, v in history_p1.history.items()}
    for k, v in history_p2.history.items():
        combined[k] = combined.get(k, []) + list(v)

    epochs = range(1, len(combined["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, ["accuracy", "auc", "loss"]):
        ax.plot(epochs, combined[metric],           label=f"Train {metric}")
        ax.plot(epochs, combined[f"val_{metric}"],  label=f"Val {metric}")
        ax.axvline(len(history_p1.history["loss"]) + 0.5,
                   color="grey", linestyle="--", linewidth=0.8,
                   label="Phase 1→2")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "checkpoints/training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Saved {out}")

    checkpoint("plots_saved", out)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stage_evaluate(model, test_ds):
    log.info("━" * 60)
    log.info("STAGE 8 — Test-set evaluation")
    log.info("━" * 60)

    results = model.evaluate(test_ds, verbose=1)
    metric_names = ["loss", "accuracy", "auc", "precision", "recall"]
    for name, val in zip(metric_names, results):
        log.info(f"  {name:>12} : {val:.4f}")

    # Collect predictions
    y_true, y_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    report = classification_report(y_true, y_pred,
                                   target_names=CONFIG["class_names"])
    log.info("\n" + report)

    # Save report to file
    with open("checkpoints/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CONFIG["class_names"],
                yticklabels=CONFIG["class_names"],
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    out = "checkpoints/confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Saved {out}")

    acc = results[1]
    checkpoint("evaluation", f"test_accuracy={acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Sample predictions
# ══════════════════════════════════════════════════════════════════════════════

def stage_sample_predictions(model, test_ds, n: int = 12):
    log.info("━" * 60)
    log.info("STAGE 9 — Sample predictions grid")
    log.info("━" * 60)

    images_batch, labels_batch = next(iter(test_ds))
    n = min(n, len(images_batch)) # added to adjust to smoke test batch size
    images_batch = images_batch[:n]
    labels_batch = labels_batch[:n].numpy().flatten()
    preds = model.predict(images_batch, verbose=0).flatten()

    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes] #added for smoke test

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img = images_batch[i].numpy().astype("uint8")
        ax.imshow(img)
        true_label = CONFIG["class_names"][int(labels_batch[i])]
        pred_label = CONFIG["class_names"][int(preds[i] >= 0.5)]
        conf = preds[i] if preds[i] >= 0.5 else 1 - preds[i]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.0%})",
                     color=color, fontsize=10)
        ax.axis("off")

    plt.suptitle("Sample Predictions (green=correct  red=wrong)", fontsize=13)
    plt.tight_layout()
    out = "checkpoints/sample_predictions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Saved {out}")

    checkpoint("predictions_saved", out)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.smoke_test:
        apply_smoke_test_overrides(CONFIG, user_data_dir=args.data_dir)
    elif args.data_dir:
        CONFIG["data_dir"] = args.data_dir

    log.info("═" * 60)
    log.info("  Cats vs Dogs — EfficientNetB0 Transfer Learning")
    log.info(f"  Mode: {'SMOKE TEST' if args.smoke_test else 'FULL TRAINING'}")
    log.info(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("═" * 60)

    stage_environment()                                     # STAGE 1
    train_ds, val_ds, test_ds = stage_data()               # STAGE 2
    model = stage_build_model()                             # STAGE 3

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    model = compile_model(model, CONFIG["learning_rate"], "phase1")
    model.summary(print_fn=log.info)
    history_p1 = stage_phase1(model, train_ds, val_ds)     # STAGE 4

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    model = stage_enable_fine_tuning(model, unfreeze_from=100)  # STAGE 5
    model = compile_model(model, CONFIG["fine_tune_lr"], "phase2")
    history_p2 = stage_phase2(model, train_ds, val_ds)     # STAGE 6

    # ── Outputs ───────────────────────────────────────────────────────────────
    stage_plot(history_p1, history_p2)                     # STAGE 7
    stage_evaluate(model, test_ds)                         # STAGE 8
    stage_sample_predictions(model, test_ds)               # STAGE 9

    log.info("═" * 60)
    log.info("  ALL STAGES COMPLETE")
    log.info(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("  Artefacts in checkpoints/")
    log.info("    best_model.keras")
    log.info("    epoch_weights/phase1/   (per-epoch weight snapshots)")
    log.info("    epoch_weights/phase2/   (per-epoch weight snapshots)")
    log.info("    phase1_metrics.csv  phase2_metrics.csv")
    log.info("    training_curves.png   confusion_matrix.png")
    log.info("    classification_report.txt")
    log.info("    run.log  completed_stages.json")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
