"""
app.py  —  SolarScan v3.0
──────────────────────────
Routes
  GET  /           → UI
  POST /predict    → YOLO *classification* + image preprocessing + auto Gemini call
  POST /explain_defect → on-demand Gemini explanation (label + confidence)
  GET  /health     → model + API key status

Key changes from v2
  • run_inference() now uses classification API (probs) instead of detection (.boxes)
  • preprocess_image() added — CLAHE contrast + grayscale→RGB + resize
  • /predict returns {label, confidence, low_confidence, gemini} in one shot
  • /explain_defect accepts optional confidence field
"""

import os
import io
import base64
import logging

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

# ── Load .env before importing gemini_helper ──────────────────────────────────
load_dotenv()

from gemini_helper import get_defect_explanation, is_api_key_configured  # noqa: E402
from download_model import ensure_model_downloaded  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

MODEL_PATH         = "model/best.pt"
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", "").strip()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
CONF_THRESHOLD     = 0.50   # below this → "Low confidence" + skip Gemini

os.makedirs("uploads", exist_ok=True)

# ── YOLO model (loaded once) ──────────────────────────────────────────────────
model = None

@app.before_request
def load_model_if_needed():
    global model
    if model is None and request.endpoint == 'predict':
        model = YOLO(MODEL_PATH)

def load_model() -> None:
    global model

    if not os.path.exists(MODEL_PATH) and MODEL_DOWNLOAD_URL:
        log.info("⬇️  Model missing; downloading from Google Drive")
        try:
            downloaded = ensure_model_downloaded(MODEL_PATH, MODEL_DOWNLOAD_URL)
            if downloaded:
                log.info("✅  Model downloaded to %s", MODEL_PATH)
        except Exception as exc:
            log.error("❌  Model download failed: %s", exc)

    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        log.info("✅  YOLO model loaded from %s", MODEL_PATH)
    else:
        log.warning("⚠️  Model not found at %s — place best.pt there or set MODEL_DOWNLOAD_URL.", MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# ██  IMAGE PREPROCESSING  ████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert any uploaded image into a clean, contrast-enhanced RGB array
    ready for YOLO classification.

    Pipeline
    ────────
    Step 1 ▸ Grayscale → RGB conversion
              Some solar panel images are single-channel.  YOLO expects 3-ch.

    Step 2 ▸ Resize to 224 × 224
              Standard classification input size; avoids distortion from
              variable-res uploads.

    Step 3 ▸ CLAHE contrast enhancement (per channel)
              Adaptive histogram equalisation fixes washed-out or dark images
              without clipping highlights globally.
              clip_limit=2.0 and tile 8×8 are conservative — good for
              solar-panel texture without introducing artefacts.

    Step 4 ▸ Mild Gaussian noise reduction
              kernel (3,3) removes sensor noise while preserving edges.

    Returns
    ───────
    np.ndarray  shape (224, 224, 3)  dtype uint8  colour-order RGB
    """

    # ── Step 1: Convert to grayscale ──────────────────────────────────────────
    img = pil_image.convert("RGB")                  # always 3-channel
    arr = np.array(img, dtype=np.uint8)             # H×W×3, RGB
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)    # H×W, single channel

    # ── Step 2: Resize to 224 × 224 ──────────────────────────────────────────
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)

    # ── Step 3: CLAHE contrast enhancement on grayscale ──────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Step 4: Mild Gaussian noise reduction ─────────────────────────────────
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    # ── Step 5: Convert back to 3-channel grayscale for YOLO input ──────────
    arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)   # (224, 224, 3) RGB-like

    log.info("Preprocessing done — output shape %s dtype %s", arr.shape, arr.dtype)
    return arr   # (224, 224, 3) uint8 grayscale RGB


# ══════════════════════════════════════════════════════════════════════════════
# ██  CLASSIFICATION INFERENCE  ███████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
def run_classification(pil_image: Image.Image) -> dict:
    """
    Preprocess → YOLO classify → return structured result dict.

    Returns
    ───────
    {
      "label":           str,    # top-1 class name
      "confidence":      float,  # 0–100
      "low_confidence":  bool,   # True when conf < CONF_THRESHOLD
      "preprocessed_b64": str,   # base64 PNG of the preprocessed image (for UI)
    }
    """
    if model is None:
        raise RuntimeError("YOLO model is not loaded. Check MODEL_PATH.")

    # 1. Preprocess
    proc_arr = preprocess_image(pil_image)           # (224,224,3) RGB uint8

    # 2. Encode preprocessed image for the frontend (so users can see it)
    proc_pil = Image.fromarray(proc_arr)
    buf = io.BytesIO()
    proc_pil.save(buf, format="PNG")
    proc_b64 = base64.b64encode(buf.getvalue()).decode()

    # 3. YOLO classification  ← FIXED: use .probs not .boxes
    results = model.predict(source=proc_arr, verbose=False)
    probs      = results[0].probs          # Probs object (classification only)
    probs_arr  = probs.data.cpu().numpy().astype(np.float32)
    if probs_arr.ndim == 0:
        probs_arr = np.atleast_1d(probs_arr)

    topk = np.argsort(probs_arr)[::-1][:2]
    top_predictions = [
        {
            "label":      model.names[int(idx)],
            "confidence": round(float(probs_arr[int(idx)]) * 100, 1),
        }
        for idx in topk
    ]

    top1 = top_predictions[0] if top_predictions else {"label": "", "confidence": 0.0}
    label      = top1["label"]
    confidence = top1["confidence"] / 100.0

    log.info("Classified as '%s' with confidence %.3f", label, confidence)

    return {
        "label":            label,
        "confidence":       round(confidence * 100, 1),   # percent
        "low_confidence":   confidence < CONF_THRESHOLD,
        "preprocessed_b64": proc_b64,
        "top_predictions":  top_predictions,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multiple image uploads via 'images' field.
    Processes each image independently and returns array of results.
    
    Response format:
    {
      "success": bool,
      "results": [
        {
          "label": str,
          "confidence": float (0-100),
          "low_confidence": bool,
          "preprocessed_img": base64 PNG str,
          "gemini": dict | null
        },
        ...
      ]
    }
    """
    if "images" not in request.files and "image" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    # Handle both 'images' (multiple) and 'image' (single, for backwards compatibility)
    files = request.files.getlist("images") or request.files.getlist("image")
    
    if not files or len(files) == 0:
        return jsonify({"error": "No images selected"}), 400

    results = []
    
    for file in files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            results.append({
                "success":       False,
                "label":         "Unknown",
                "confidence":    0,
                "low_confidence": True,
                "preprocessed_img": None,
                "error":         f"Unsupported file type: {file.filename}",
                "gemini":        None,
            })
            continue

        try:
            pil_image = Image.open(file.stream)
            result    = run_classification(pil_image)

            label      = result["label"]
            confidence = result["confidence"]
            low_conf   = result["low_confidence"]

            # Only classify images during batch /predict.
            # Gemini explanations are requested later via the Explain button.
            results.append({
                "success":          True,
                "label":            label,
                "confidence":       confidence,
                "low_confidence":   low_conf,
                "preprocessed_img": f"data:image/png;base64,{result['preprocessed_b64']}",
                "error":            None,
                "gemini":           None,
            })

        except RuntimeError as e:
            log.error("Runtime error in classification: %s", e)
            results.append({
                "success":       False,
                "label":         "Unknown",
                "confidence":    0,
                "low_confidence": True,
                "preprocessed_img": None,
                "error":         str(e),
                "gemini":        None,
            })
        except Exception as e:
            log.exception("Inference error")
            results.append({
                "success":       False,
                "label":         "Unknown",
                "confidence":    0,
                "low_confidence": True,
                "preprocessed_img": None,
                "error":         f"Inference failed: {str(e)}",
                "gemini":        None,
            })

    if len(results) == 0:
        return jsonify({"error": "No valid images to process"}), 400

    return jsonify({
        "success": True,
        "results": results,
    })


@app.route("/explain_defect", methods=["POST"])
def explain_defect():
    """
    POST JSON: {"defect_name": "crack", "confidence": 87.3}
    On-demand Gemini explanation (called from UI's Explain button).
    """
    try:
        body = request.get_json(force=True)
    except Exception as e:
        log.warning("JSON parse error in /explain_defect: %s", e)
        body = None
    
    if not body:
        log.warning("No JSON body received in /explain_defect. Content-Type: %s", request.content_type)
        return jsonify({"error": "Request body must be valid JSON with 'defect_name'"}), 400
    
    if "defect_name" not in body:
        log.warning("Missing 'defect_name' in request body: %s", body)
        return jsonify({"error": "Request body must include 'defect_name'"}), 400

    defect_name = str(body.get("defect_name", "")).strip()
    confidence  = float(body.get("confidence", 0.0))

    if not defect_name:
        log.warning("Empty defect_name provided")
        return jsonify({"error": "'defect_name' cannot be empty"}), 400

    log.info("Explaining defect: '%s' (confidence: %.1f%%)", defect_name, confidence)
    result = get_defect_explanation(defect_name, confidence)
    return jsonify(result), 200


@app.route("/health")
def health():
    model_info = {
        "status":            "ok",
        "model_loaded":      model is not None,
        "model_path":        MODEL_PATH,
        "gemini_configured": is_api_key_configured(),
    }
    
    if model is not None:
        model_info["model_classes"] = model.names
        model_info["num_classes"] = len(model.names)
    
    return jsonify(model_info)


# Load the model when the module is imported, so WSGI deployments can start ready.
load_model()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from waitress import serve
        print("🚀 Starting with Waitress (production server)...")
        serve(app, host="0.0.0.0", port=5000, threads=8)
    except ImportError:
        print("⚠️  Waitress not installed. Using Flask dev server.")
        app.run(debug=False, host="0.0.0.0", port=5000)
