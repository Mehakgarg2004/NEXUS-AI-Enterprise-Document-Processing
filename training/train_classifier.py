"""Train the TensorFlow document classifier and track it with MLflow.

Usage:
    python -m training.train_classifier                 # synthetic data
    python -m training.train_classifier --data my.csv   # your labelled CSV (columns: text,label)

The trained model is saved to MODEL_DIR (default: models/doc_classifier)
and logged to MLflow (params, per-epoch metrics, classification report,
confusion matrix and the model artifact).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from nexus.classifier import DOC_TYPES, _clean  # noqa: E402
from nexus.config import Config  # noqa: E402
from training.synthetic import generate  # noqa: E402


def load_csv(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["text"], r["label"]) for r in csv.DictReader(f) if r.get("text") and r.get("label")]


def build_model(train_texts, n_classes: int, vocab: int, seq_len: int, embed_dim: int, dropout: float):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab, output_mode="int", output_sequence_length=seq_len, ngrams=None
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(256))

    inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inp)
    x = tf.keras.layers.Embedding(vocab, embed_dim, mask_zero=True)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class MLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with columns text,label (default: synthetic)")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=400)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--out", default=Config.MODEL_DIR)
    ap.add_argument("--experiment", default="nexus-doc-classifier")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)
    data = load_csv(args.data) if args.data else generate(args.per_class, seed=7, noise=args.noise)
    labels = sorted({l for _, l in data}) if args.data else DOC_TYPES
    idx = {l: i for i, l in enumerate(labels)}

    random.Random(42).shuffle(data)
    n = len(data)
    n_test, n_val = int(n * 0.15), int(n * 0.15)
    test, val, train = data[:n_test], data[n_test : n_test + n_val], data[n_test + n_val :]

    def xy(rows):
        return tf.constant([_clean(t) for t, _ in rows]), np.array([idx[l] for _, l in rows])

    Xtr, ytr = xy(train)
    Xva, yva = xy(val)
    Xte, yte = xy(test)

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**vars(args), "n_train": len(train), "n_val": len(val), "n_test": len(test), "labels": ",".join(labels)})

        model = build_model(Xtr, len(labels), args.vocab, args.seq_len, args.embed_dim, args.dropout)
        model.fit(
            Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch_size, verbose=2,
            callbacks=[MLflowCallback(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        )

        probs = model.predict(Xte, verbose=0)
        pred = probs.argmax(1)
        acc = float((pred == yte).mean())
        report = classification_report(yte, pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
        mlflow.log_metrics({"test_accuracy": acc, "test_macro_f1": report["macro avg"]["f1-score"]})
        for l in labels:
            mlflow.log_metric(f"f1_{l}", report[l]["f1-score"])

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model.keras")
        (out / "labels.json").write_text(json.dumps(labels))
        meta = {"mlflow_run_id": run.info.run_id, "test_accuracy": acc, "macro_f1": report["macro avg"]["f1-score"]}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "classification_report.json").write_text(json.dumps(report, indent=2))
            cm = confusion_matrix(yte, pred, labels=list(range(len(labels))))
            Path(tmp, "confusion_matrix.json").write_text(json.dumps({"labels": labels, "matrix": cm.tolist()}))
            mlflow.log_artifacts(tmp, "evaluation")
        mlflow.log_artifacts(str(out), "model")

        print(f"\nTest accuracy: {acc:.4f}  macro-F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Model saved to {out}  |  MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
