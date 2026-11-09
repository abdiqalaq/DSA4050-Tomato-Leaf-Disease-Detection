
print("Python script is running!")
import tensorflow_datasets as tfds

# Load dataset metadata without downloading images yet
builder = tfds.builder("plant_village")

info = builder.info

print("=" * 55)
print("PLANTVILLAGE DATASET INFORMATION")
print("=" * 55)

print(f"Dataset name: {info.name}")
print(f"Total images: {info.splits.total_num_examples}")
print(f"Number of classes: {info.features['label'].num_classes}")

# Get all class names
class_names = info.features["label"].names

print("\nTOMATO LEAF CATEGORIES")
print("-" * 55)

tomato_classes = [
    name for name in class_names
    if name.startswith("Tomato___")
]

for index, name in enumerate(tomato_classes, start=1):
    print(f"{index}. {name}")

print(f"\nTotal tomato categories: {len(tomato_classes)}")
