
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# Paths
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Load CSV files
train_df = pd.read_csv(SPLITS / "train.csv")
val_df = pd.read_csv(SPLITS / "val.csv")

class_names = sorted(train_df["label"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}

train_df["class_id"] = train_df["label"].map(class_to_index)
val_df["class_id"] = val_df["label"].map(class_to_index)

print("Classes:", class_names)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))

# Image loading and preprocessing
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(df, training=False):
    paths = [
        str(ROOT / path)
        for path in df["image_path"]
    ]
    labels = df["class_id"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

# Calculate class weights for imbalanced classes
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_df["class_id"]
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

# Data augmentation applied only during training
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

# Custom CNN architecture
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    augmentation,

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model and stop when validation stops improving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / "custom_cnn_best.keras"),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.CSVLogger(
        str(RESULTS / "custom_cnn_history.csv")
    )
]

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save class names for dashboard use
pd.DataFrame({"class_name": class_names}).to_csv(
    RESULTS / "class_names.csv",
    index=False
)

print("\nCNN training finished!")
print("Best model saved to models/custom_cnn_best.keras")
print("Training history saved to results/custom_cnn_history.csv")
