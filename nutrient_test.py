from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("nutrient_model.h5")


# Define class labels
class_labels = ['Nitrogen', 'Phosphorus', 'Potassium']

# Load an image to test
img_path = r"C:\Users\SSN\OneDrive - SSN-Institute\Desktop\untitled-306.JPG"  # Replace with your image path
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)
class_idx = np.argmax(pred)
print("Predicted nutrient deficiency:", class_labels[class_idx])
