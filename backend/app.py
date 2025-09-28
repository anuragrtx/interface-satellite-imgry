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
# Ensure your Keras model file is in the same directory as this script
MODEL_FILENAME = "satellite_standard_unet_100epochs.hdf5"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

# ====== LOAD MODEL ======
# Check if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please make sure '{MODEL_FILENAME}' is in the same directory.")

print(f"Loading model from {MODEL_PATH} ...")
# Load the model. `compile=False` is used when the model is only for inference.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

# ====== COLOR MAP ======
# The order is crucial and must correspond to the model's class indices.
# Order: 0:Building, 1:Land, 2:Road, 3:Vegetation, 4:Water, 5:Unlabeled
CLASS_COLORS = [
    (60, 16, 152),      # 0: Building (#3C1098)
    (132, 41, 246),     # 1: Land (#8429F6)
    (110, 193, 228),    # 2: Road (#6EC1E4)
    (254, 221, 58),     # 3: Vegetation (#FEDD3A)
    (226, 169, 41),     # 4: Water (#E2A929)
    (155, 155, 155)     # 5: Unlabeled (#9B9B9B)
]

# ====== PREDICTION ROUTE (Enhanced for Debugging) ======
@app.route("/predict", methods=["POST"])
def predict():
    # This is the very first thing that should run when the route is called.
    print("\n--- ✅ /predict route was hit ---") 
    
    if "image" not in request.files:
        print("--- ❌ ERROR: 'image' key not found in request.files ---")
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        print("--- 1. Entering a TRY block to process the image. ---")
        
        img_file = request.files["image"]
        print(f"--- 2. File received: '{img_file.filename}' ---")

        original_img = Image.open(img_file).convert("RGB")
        print("--- 3. Image successfully opened with PIL. ---")

        img_resized = original_img.resize((256, 256))
        print("--- 4. Image resized to 256x256. ---")

        img_arr = np.array(img_resized) / 255.0
        img_arr = np.expand_dims(img_arr, 0)
        print(f"--- 5. Image converted to numpy array with shape: {img_arr.shape} ---")

        # This is often where errors happen if the input shape is wrong
        print("--- 6. Calling model.predict()... ---")
        pred_mask = model.predict(img_arr)[0]
        print("--- 7. Model prediction successful! ---")

        mask_classes = np.argmax(pred_mask, axis=-1)
        print("--- 8. Argmax calculated to find predicted classes. ---")

        unique_classes, class_counts = np.unique(mask_classes, return_counts=True)
        print("\n✨✨✨ DIAGNOSTIC RESULT ✨✨✨")
        print(f"--> Predicted Classes: {dict(zip(unique_classes, class_counts))}")
        print("✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨\n")

        # Create the color mask from the class predictions
        color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in enumerate(CLASS_COLORS):
            color_mask[mask_classes == class_id] = color
        print("--- 9. Color mask created. ---")

        # Convert the mask back to an image and resize
        mask_img = Image.fromarray(color_mask).resize(original_img.size, Image.NEAREST)

        # Encode the final image to a base64 string
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print("--- 10. Mask encoded to Base64. Sending response... ---")

        return jsonify({
            "mask": f"data:image/png;base64,{mask_base64}"
        })

    except Exception as e:
        # This block will catch any error inside the `try` block and print it.
        print(f"\n--- ❌❌❌ AN ERROR OCCURRED ❌❌❌ ---")
        print(f"--- Exception Type: {type(e).__name__}")
        print(f"--- Exception Message: {e}")
        print("--- Full Traceback: ---")
        traceback.print_exc() # This gives the most detailed error report
        print("------------------------------------")
        return jsonify({"error": "An internal error occurred. Check server logs for details."}), 500


if __name__ == "__main__":
    # Run the Flask app on port 5001, accessible from any IP
    app.run(host="0.0.0.0", port=5001)