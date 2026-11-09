
import tensorflow_datasets as tfds

print("=" * 55)
print("DOWNLOADING PLANTVILLAGE DATASET")
print("=" * 55)

# Download and prepare the dataset
builder = tfds.builder(
    "plant_village",
    data_dir="data/tfds"
)

builder.download_and_prepare()

# Read the prepared dataset information
info = builder.info

print("\nDATASET DOWNLOAD COMPLETED")
print(f"Total images: {info.splits.total_num_examples}")
print(f"Number of classes: {info.features['label'].num_classes}")

# Identify tomato categories
class_names = info.features["label"].names

tomato_classes = [
    name for name in class_names
    if name.startswith("Tomato___")
]

print("\nTOMATO LEAF CATEGORIES")
for name in tomato_classes:
    print(name)

print(f"\nTotal tomato categories: {len(tomato_classes)}")
