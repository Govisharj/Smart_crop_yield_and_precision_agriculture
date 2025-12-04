import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("plant_soil_combined_cnn.h5")

def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)[0][0]

    if result > 0.5:
        return "WET – no need to water"
    else:
        return "DRY – water required"

print(predict(r"C:\Users\Risho\Downloads\images (2).jpeg"))
