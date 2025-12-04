import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report
import numpy as np

# Paths
TRAIN_PATH = r"C:\Users\Risho\OneDrive\Desktop\new_model\Rice_Leaf_Disease\Rice_Leaf_Diease\train"
TEST_PATH  = r"C:\Users\Risho\OneDrive\Desktop\new_model\Rice_Leaf_Disease\Rice_Leaf_Diease\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data generators
train_gen = ImageDataGenerator(rescale=1/255.0)
test_gen  = ImageDataGenerator(rescale=1/255.0)

train = train_gen.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"

)

test = test_gen.flow_from_directory(
    TEST_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"

)

num_classes = train.num_classes

# Load EfficientNetB0 (fastest + high accuracy)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


# Freeze base model (super fast training)
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train (VERY FAST)
history = model.fit(train, epochs=10, validation_data=test)

# Evaluate
loss, acc = model.evaluate(test)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

# Predictions for classification report
test.reset()
preds = model.predict(test)
pred_labels = np.argmax(preds, axis=1)

true_labels = test.classes

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=list(test.class_indices.keys())))

model.save("rice_leaf_effnet.keras")
