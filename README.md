
# Tomato Leaf Disease Detection Using Deep Learning

**Course:** DSA4050_XA – Deep Learning  
**Project:** Term Paper and Practical Implementation  
**Institution:** United States International University–Africa

## 1. Project Overview

This project applies deep learning to identify diseases affecting tomato leaves from images. Two image classification models were developed and evaluated:

- A Custom Convolutional Neural Network (CNN)
- MobileNetV2 using transfer learning

The models were trained and evaluated using tomato leaf images from the PlantVillage dataset. An interactive Streamlit dashboard was developed to visualize model performance and predict diseases from uploaded leaf images.

## 2. Objectives

1. Prepare a dataset of tomato leaf images for deep learning.
2. Train a custom CNN for disease classification.
3. Apply transfer learning using MobileNetV2.
4. Compare the models using accuracy, precision, recall, and F1-score.
5. Develop an interactive dashboard for image-based disease prediction.

## 3. Dataset

**Dataset:** PlantVillage – Tomato Leaf Images

The dataset contains images of healthy and diseased tomato leaves representing different disease categories.

Dataset preparation includes organizing the images, creating training, validation, and testing splits, and preprocessing images for model training.

The full image dataset is excluded from this repository because of its size.

The `data/splits/` directory contains the training, validation, and testing split files.

Dataset-related scripts:

- `src/01_explore_dataset.py`
- `src/02_download_dataset.py`
- `src/03_prepare_dataset.py`

## 4. Deep Learning Models

### Custom CNN

A convolutional neural network was trained to learn visual features from tomato leaf images.

Training script:

`src/04_train_cnn.py`

Saved model:

`models/custom_cnn_best.keras`

### MobileNetV2

MobileNetV2 was used as a transfer-learning architecture to improve classification performance.

Training script:

`src/05_train_mobilenet.py`

Saved model:

`models/mobilenetv2_best.keras`

## 5. Model Evaluation

Both models were evaluated using classification metrics and confusion matrices.

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 70.70% | 89.65% |
| Precision | 68.48% | 87.30% |
| Recall | 70.91% | 88.70% |
| F1-score | 67.20% | 87.66% |

MobileNetV2 achieved the highest classification accuracy, outperforming the Custom CNN by 18.95 percentage points.

Evaluation script:

`src/06_evaluate_models.py`

Detailed results are available in the `results/` directory.

## 6. Interactive Streamlit Dashboard

The dashboard allows users to explore the trained model results and perform image-based tomato leaf disease prediction.

The application supports uploading a tomato leaf image and displaying a predicted disease class with model confidence.

### Running the Dashboard

Clone the repository:

```bash
git clone https://github.com/abdiqalaq/DSA4050-Tomato-Leaf-Disease-Detection.git
```

Enter the project directory:

```bash
cd DSA4050-Tomato-Leaf-Disease-Detection
```

Create and activate a Python virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Open the local address displayed in the terminal, normally:

`http://localhost:8501`

The trained model files are included in the repository, so retraining is not required to run the dashboard.

**Note:** Running the training and evaluation scripts requires the original PlantVillage images to be available locally in the directory structure expected by the scripts.

## 7. Project Structure

```text
DSA4050-Tomato-Leaf-Disease-Detection/
├── dashboard/
│   └── app.py
├── data/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── custom_cnn_best.keras
│   └── mobilenetv2_best.keras
├── presentation/
│   └── Tomato_Leaf_Disease_Presentation.pptx
├── results/
│   ├── model_comparison.csv
│   ├── class_names.csv
│   ├── custom_cnn_history.csv
│   ├── mobilenetv2_history.csv
│   └── confusion_matrix_and_classification_files
├── src/
│   ├── 01_explore_dataset.py
│   ├── 02_download_dataset.py
│   ├── 03_prepare_dataset.py
│   ├── 04_train_cnn.py
│   ├── 05_train_mobilenet.py
│   └── 06_evaluate_models.py
├── requirements.txt
└── README.md
```

## 8. Conclusion

The results demonstrate that deep learning can support automated tomato leaf disease classification.

MobileNetV2 achieved better performance than the Custom CNN, showing the benefits of transfer learning for this image classification task.

The Streamlit dashboard provides an interactive interface for exploring the results and testing predictions on uploaded images.

## 9. Limitations

- The models were evaluated using the selected PlantVillage dataset.
- Performance on real-world field images may differ because of lighting, backgrounds, and image quality.
- Predictions should not be treated as definitive agricultural diagnoses.
- The full image dataset must be obtained separately for retraining.

## 10. Repository

https://github.com/abdiqalaq/DSA4050-Tomato-Leaf-Disease-Detection
