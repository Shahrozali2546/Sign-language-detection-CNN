import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ----- Constants -----#
IMG_SIZE = 96 
MODEL_PATH = "asl_best_model.keras"


model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Success: Latest TensorFlow Loaded the Model!")
    except Exception as e:
        print(f" Error after update: {e}")
else:
    print(f" Warning: {MODEL_PATH} not found!")


labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
        
    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid format"}), 400

    # Model Prediction
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    
    class_id = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": labels_list[class_id],
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
