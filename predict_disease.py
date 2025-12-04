import tensorflow as tf
import numpy as np
import cv2

# --- SETTINGS ---
IMG_SIZE = (224, 224)
MODEL_PATH = r"C:\Users\Risho\OneDrive\Desktop\new_model\rice_leaf_effnet.keras"    # Change if needed
IMAGE_PATH = r"C:\Users\Risho\Downloads\download.jpg"    # your input image

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- LOAD CLASS NAMES ---
# If your training generator had classes like:
# train.class_indices = {'brown_spot':0, 'leaf_blast':1, ...}
# create the SAME order here:

class_names = ["bacterial_leaf_blight", "brown_spot", "leaf_smut", "rice_hispa"]  
# <-- Change names based on your dataset folders (important)

# --- IMAGE PREPROCESSING ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# --- PREDICT ---
img = preprocess_image(IMAGE_PATH)
pred = model.predict(img)

pred_class = np.argmax(pred)
confidence = np.max(pred)

print("\n========= PREDICTION RESULT =========")
print(f"Predicted Class : {class_names[pred_class]}")
print(f"Confidence      : {confidence * 100:.2f}%")
print("=====================================\n")
