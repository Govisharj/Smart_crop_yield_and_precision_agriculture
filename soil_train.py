import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# =========================
# Path to dataset
# =========================
train_dir = r"D:\SEM5\SIH\usedForpaper\Soil_Moisture_Dataset"

# =========================
# Data Preprocessing & Augmentation
# =========================
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_set = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_set = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# =========================
# Compute Class Weights for Imbalanced Data
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_set.classes),
    y=train_set.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =========================
# Build Transfer Learning Model (MobileNetV2)
# =========================
base_model = MobileNetV2(
    include_top=False,
    input_shape=(128, 128, 3),
    weights='imagenet'
)
base_model.trainable = False  # freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# =========================
# Compile Model
# =========================
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# =========================
# Callbacks
# =========================
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)

# =========================
# Train Model
# =========================
history = model.fit(
    train_set,
    epochs=30,
    validation_data=val_set,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# =========================
# Evaluate Model
# =========================
val_set.reset()
val_loss, val_accuracy = model.evaluate(val_set)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# =========================
# Predictions & Metrics
# =========================
predictions = model.predict(val_set)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = val_set.classes

# Classification report
print("\nClassification Report:")
class_labels = list(val_set.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_soil_validation.png')
plt.show()

# =========================
# Plot Accuracy & Loss
# =========================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_metrics_soil.png')
plt.show()

# =========================
# Save Model
# =========================
model.save("soil_mobilenetv2_model.h5")
print("✅ Model trained and saved as soil_mobilenetv2_model.h5")


'''
Validation Accuracy: 0.6718
Validation Loss: 0.5838
9/9 ━━━━━━━━━━━━━━━━━━━━ 19s 2s/step

Classification Report:
              precision    recall  f1-score   support

         dry       0.60      0.57      0.59       161
         wet       0.35      0.38      0.36        98

    accuracy                           0.50       259
   macro avg       0.48      0.47      0.47       259
weighted avg       0.51      0.50      0.50       259

'''