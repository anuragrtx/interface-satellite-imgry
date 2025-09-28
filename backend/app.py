from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, base64
import traceback

app = Flask(__name__)
CORS(app)

# ====== CONFIG ======
# Model filename (keep in same folder as this script)
MODEL_FILENAME = "satellite_standard_unet_100epochs.hdf5"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

# ====== LOAD MODEL ======
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please make sure '{MODEL_FILENAME}' is in the same directory.")

print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

# ====== COLOR MAP ======
CLASS_COLORS = [
    (60, 16, 152),      # 0: Building
    (132, 41, 246),     # 1: Land
    (110, 193, 228),    # 2: Road
    (254, 221, 58),     # 3: Vegetation
    (226, 169, 41),     # 4: Water
    (155, 155, 155)     # 5: Unlabeled
]

# ====== PREDICTION ROUTE ======
@app.route("/predict", methods=["POST"])
def predict():
    print("\n--- ✅ /predict route was hit ---") 

    if "image" not in request.files:
        print("--- ❌ ERROR: 'image' key not found in request.files ---")
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        img_file = request.files["image"]
        print(f"--- File received: '{img_file.filename}' ---")

        original_img = Image.open(img_file).convert("RGB")
        img_resized = original_img.resize((256, 256))
        img_arr = np.expand_dims(np.array(img_resized) / 255.0, 0)
        print(f"Image prepared for model with shape: {img_arr.shape}")

        pred_mask = model.predict(img_arr)[0]
        mask_classes = np.argmax(pred_mask, axis=-1)
        print(f"Predicted classes: {np.unique(mask_classes, return_counts=True)}")

        # Create color mask
        color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in enumerate(CLASS_COLORS):
            color_mask[mask_classes == class_id] = color

        mask_img = Image.fromarray(color_mask).resize(original_img.size, Image.NEAREST)

        # Encode mask to base64
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({"mask": f"data:image/png;base64,{mask_base64}"})

    except Exception as e:
        print(f"\n--- ❌ AN ERROR OCCURRED ---")
        print(f"Type: {type(e).__name__}, Message: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred. Check server logs for details."}), 500


if __name__ == "__main__":
    # Use Render-provided PORT if available, else default to 5001
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)