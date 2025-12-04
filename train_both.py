import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report

# Image size
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load dataset
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    r"C:\Users\Risho\OneDrive\Desktop\new_model\images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    r"C:\Users\Risho\OneDrive\Desktop\new_model\images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")    # wet or dry
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train, epochs=20, validation_data=val)
val_loss, val_accuracy = model.evaluate(val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Predict on validation data to generate classification report
val.reset()
preds = model.predict(val)
pred_labels = (preds > 0.5).astype(int).reshape(-1)

# True labels from validation generator
true_labels = val.classes

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=list(val.class_indices.keys())))

model.save("plant_soil_combined_cnn.h5")

''': 0.7136 - loss: 0.8854
Validation Accuracy: 0.7136
Validation Loss: 0.8854
76/76 ━━━━━━━━━━━━━━━━━━━━ 2997s 98ms/step    
Classification Report:
              precision    recall  f1-score   support

         dry       0.50      0.53      0.52       600
         wet       0.51      0.48      0.49       601

    accuracy                           0.51      1201
   macro avg       0.51      0.51      0.51      1201
weighted avg       0.51      0.51      0.51      1201

WARNING:absl:You are saving your model as an HDF5 file via model.save() or keras.saving.save_model(model). This file format is considered legacy. We recommend using instead the native Keras format, e.g. model.save('my_model.keras') or keras.saving.save_model(model, 'my_model.keras').
PS C:\Users\Risho\OneDrive\Desktop\new_model>'''