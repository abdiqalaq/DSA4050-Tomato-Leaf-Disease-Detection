
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "plantvillage" / "raw" / "color"
OUTPUT_DIR = PROJECT_DIR / "data" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect tomato leaf images and their labels
records = []

for class_dir in sorted(DATA_DIR.glob("Tomato___*")):
    if class_dir.is_dir():
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({
                    "image_path": str(image_path.relative_to(PROJECT_DIR)),
                    "label": class_dir.name
                })

df = pd.DataFrame(records)

print(f"Total images: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")

# First split: 80% training, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Second split: 10% validation, 10% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Save CSV files without copying images
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nDataset split completed successfully!")
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images: {len(test_df)}")

print("\nCSV files saved in data/splits/")
