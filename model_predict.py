import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

def load_model(model_path='transfer_learning_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_class_indices(json_path='class_indices.json'):
    with open(json_path, 'r') as f:
        return json.load(f)

def predict_image(model, img_path, class_indices):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    reverse_map = {v: k for k, v in class_indices.items()}
    predicted_class = reverse_map[predicted_index]
    return predicted_class, confidence

def display_prediction(img_path, class_name, confidence):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"Prediction: {class_name} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

def main():
    model = load_model()
    class_indices = load_class_indices()

    img_path = input("Enter image path: ").strip()
    if not os.path.exists(img_path):
        print("Image not found.")
        return

    class_name, confidence = predict_image(model, img_path, class_indices)
    print(f"\nPrediction: {class_name}\nConfidence: {confidence:.2%}")
    display_prediction(img_path, class_name, confidence)

if __name__ == "__main__":
    main()
