# ☀️ SolarScan v2.0 — Solar Panel Defect Detection + AI Analysis

A Flask web app that combines YOLO defect detection with Gemini AI-powered explanations.

## Project Structure

```
solar_defect_app/
├── app.py                  ← Flask backend (detection + Gemini endpoints)
├── gemini_helper.py        ← ✨ NEW: Gemini API integration
├── requirements.txt        ← Python dependencies
├── .env                    ← ⚠️ Add your GEMINI_API_KEY here
├── .env.example            ← Template for .env
├── README.md
├── model/
│   └── best.pt             ← ⚠️ Place your trained YOLO model here
└── templates/
    └── index.html          ← Frontend UI with Gemini explanation UI
```

## Quick Start

### 1. Add your YOLO model
```bash
cp /path/to/your/best.pt solar_defect_app/model/best.pt
```

### 2. Get a free Gemini API key
- Go to: https://aistudio.google.com/app/apikey
- Create a key (free, no credit card needed)

### 3. Set your API key
Edit `.env`:
```
GEMINI_API_KEY=AIzaSy...your_actual_key...
```

### 4. Enable auto-download on Render (optional)
If you want Render to fetch the model automatically instead of committing `model/best.pt`, add:
```
MODEL_DOWNLOAD_URL=https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
```

If `model/best.pt` is missing at startup, the app will download it from the configured Google Drive link.

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```
Open: **http://localhost:5000**

---

## Features

| Feature | Description |
|---|---|
| Image upload | Drag-and-drop or click to browse |
| YOLO detection | Runs your trained model, draws bounding boxes |
| Detection list | Shows all detected defects with confidence scores |
| ✦ Explain button | Per-defect Gemini AI analysis |
| ✦ Explain All | Analyse every detected defect in sequence |
| Download | Save the annotated result image |
| Status pills | Live model + Gemini API key status in header |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/predict` | YOLO inference — multipart image upload |
| POST | `/explain_defect` | Gemini explanation — JSON `{"defect_name":"..."}` |
| GET | `/health` | Model + Gemini key status |

### `/explain_defect` response
```json
{
  "success": true,
  "defect": "Bird dropping",
  "explanation": "...",
  "sections": {
    "what_it_is": "...",
    "causes": "...",
    "impact": "...",
    "solution": "...",
    "urgency": "Medium — ..."
  },
  "error": null
}
```

---

## Files Modified / Added (v2.0)

| File | Change |
|---|---|
| `gemini_helper.py` | **NEW** — Gemini API client, prompt, response parser |
| `app.py` | Added `dotenv` load, Gemini import, `/explain_defect` endpoint, updated `/health` |
| `templates/index.html` | Added Gemini status pill, per-defect Explain buttons, AI drawer UI, Explain All button |
| `requirements.txt` | Added `requests`, `python-dotenv` |
| `.env` / `.env.example` | **NEW** — API key configuration |

