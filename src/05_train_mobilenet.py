
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Training settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Load existing dataset splits
train_df = pd.read_csv(SPLITS / "train.csv")
val_df = pd.read_csv(SPLITS / "val.csv")

class_names = sorted(train_df["label"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}

train_df["class_id"] = train_df["label"].map(class_to_index)
val_df["class_id"] = val_df["label"].map(class_to_index)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", len(class_names))

# Load images
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(df, training=False):
    paths = [str(ROOT / p) for p in df["image_path"]]
    labels = df["class_id"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            len(df),
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

# Class weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_df["class_id"]
)

class_weights = {
    i: float(w) for i, w in enumerate(weights)
}

# Image augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
], name="augmentation")

# Pretrained MobileNetV2 backbone
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build transfer learning model
inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / "mobilenetv2_best.keras"),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.CSVLogger(
        str(RESULTS / "mobilenetv2_history.csv")
    )
]

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\nMobileNetV2 training completed!")
print("Best model: models/mobilenetv2_best.keras")
print("History: results/mobilenetv2_history.csv")
