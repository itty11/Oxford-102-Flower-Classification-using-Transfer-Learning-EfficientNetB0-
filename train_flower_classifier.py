import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIG
base_dir = "archive/dataset"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32
epochs = 20

# DATA LOADERS
train_gen = ImageDataGenerator(rescale=1./255)
valid_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse"
)

valid_data = valid_gen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse"
)

# Detect if test directory has subfolders (classified) or not
test_subdirs = [f.path for f in os.scandir(test_dir) if f.is_dir()]
if len(test_subdirs) > 0:
    print(f"✅ Detected {len(test_subdirs)} labeled test folders.")
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )
    labeled_test = True
else:
    print("⚠️ No labeled folders found in test directory. Treating as unlabeled images.")
    test_data = test_gen.flow_from_directory(
        base_dir,
        classes=['test'],
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    labeled_test = False

num_classes = len(train_data.class_indices)
print(f"Number of classes: {num_classes}")

# MODEL DEFINITION
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
base_model.trainable = False  # freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAINING
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs
)

# SAVE MODEL
os.makedirs("models", exist_ok=True)
model.save("models/flower_classifier.keras")
print("✅ Model saved to models/flower_classifier.keras")

# EVALUATION / PREDICTION
if labeled_test:
    test_loss, test_acc = model.evaluate(test_data)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_true = test_data.classes

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=list(test_data.class_indices.keys())
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=list(test_data.class_indices.keys()),
                yticklabels=list(test_data.class_indices.keys()), cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

else:
    print("\n⚙️ Running predictions on unlabeled test images...")
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    inv_class_map = {v: k for k, v in train_data.class_indices.items()}
    predicted_labels = [inv_class_map[c] for c in predicted_classes]

    # Show first 10 predictions
    print("\nSample Predictions:")
    for i, label in enumerate(predicted_labels[:10]):
        print(f"Image {i+1}: Predicted class → {label}")