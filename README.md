# Oxford-102-Flower-Classification-using-Transfer-Learning-EfficientNetB0-
This project implements an end-to-end Image Classification pipeline using the Oxford 102 Flower Dataset, leveraging Transfer Learning (EfficientNetB0) for robust feature extraction and fine-tuning.   The trained model is deployed via a Streamlit web app for real-time image predictions.


# Project Overview

- **Dataset**: Oxford 102 Flowers
- **Architecture**: Transfer Learning with `EfficientNetB0` (pretrained on ImageNet)  
- **Framework**: TensorFlow / Keras  
- **Interface**: Streamlit web app for interactive predictions  
- **Goal**: Classify flower species (102 categories) from user-uploaded images

# Download Dataset

Get the Oxford 102 Flower Dataset from Kaggle:

https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/data

Extract it under the project folder:

archive/

|──dataset/

  ├── train/

  ├── valid/

  └── test/

Each folder should contain subfolders for each flower class (e.g., 1, 2, 3, …, 102).

The script also auto-detects if the test folder doesn’t contain subfolders.

Training the Model

Run the training script:

py -3.11 train_flower_classifier.py

# Features:

Automatically detects number of classes

Applies data augmentation (rotation, zoom, flip, etc.)

Fine-tunes EfficientNetB0

Saves model as models/flower_model.keras

Evaluates accuracy, loss, and classification report

# Model Evaluation & Visualization

The script generates:

Accuracy and loss plots using matplotlib

Classification metrics using scikit-learn

Confusion matrix heatmaps for detailed evaluation

Example visualizations:

Training vs Validation Accuracy

Training vs Validation Loss

Top Misclassified Classes

Running the Streamlit App

Once training is complete, run the Streamlit app:

streamlit run app.py


# App Features:

Upload any flower image (.jpg, .jpeg, .png)

Displays the image

Predicts flower name (or class number)

Shows confidence bar chart (Top-5 predictions)

Automatically detects missing cat_to_name.json and uses numeric labels instead

# Future Improvements

Add Grad-CAM visualization for model interpretability

Use mixed precision training for faster convergence

Integrate Hugging Face Spaces deployment

Add data augmentation customization via UI

# Author

Ittyavira C Abraham
🎓 MCA (AI), Amrita Vishwa Vidyapeetham
💡 Focus: AI/ML, Computer Vision, and Generative AI
