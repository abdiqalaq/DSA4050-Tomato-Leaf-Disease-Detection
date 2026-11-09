
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load test dataset
test_df = pd.read_csv(SPLITS / "test.csv")
class_names = sorted(test_df["label"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}

test_df["class_id"] = test_df["label"].map(class_to_index)

paths = [str(ROOT / p) for p in test_df["image_path"]]
y_true = test_df["class_id"].to_numpy(dtype=np.int32)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


test_ds = tf.data.Dataset.from_tensor_slices((paths, y_true))
test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model configurations
model_paths = {
    "Custom_CNN": MODELS / "custom_cnn_best.keras",
    "MobileNetV2": MODELS / "mobilenetv2_best.keras",
}

results = []

for model_name, model_path in model_paths.items():
    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(model_path)

    # Custom CNN expects pixels scaled to [0, 1].
    # MobileNetV2 contains its own preprocessing layer.
    if model_name == "Custom_CNN":
        evaluation_ds = test_ds.map(
            lambda x, y: (x / 255.0, y)
        )
    else:
        evaluation_ds = test_ds

    probabilities = model.predict(evaluation_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1,
    })

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS / f"{model_name}_classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    pd.DataFrame(
        cm, index=class_names, columns=class_names
    ).to_csv(RESULTS / f"{model_name}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.savefig(
        RESULTS / f"{model_name}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

# Save overall comparison
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(
    RESULTS / "model_comparison.csv",
    index=False
)

print("\nFinal model comparison:")
print(comparison_df.to_string(index=False))
print("\nEvaluation files saved in results/")
