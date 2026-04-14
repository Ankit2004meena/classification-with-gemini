"""
gemini_helper.py  —  SolarScan v3.0
─────────────────────────────────────
Changes from v2
  • get_defect_explanation() now accepts confidence (float, 0-100) as 2nd arg
  • Prompt updated to include confidence value for richer, context-aware answers
  • is_api_key_configured() unchanged
  • All error handling unchanged
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={{api_key}}"
)

# ── Prompt template (updated to include confidence) ───────────────────────────
PROMPT_TEMPLATE = """You are an expert solar panel engineer and maintenance specialist.

A YOLO classification model has analysed a solar panel image and returned:
  Defect  : {defect_name}
  Confidence: {confidence}%

Please provide a detailed, structured explanation using EXACTLY this format:

**WHAT IT IS**
<describe the defect clearly in 3-5 sentences>

**CAUSES**
- <cause 1>
- <cause 2>
- <cause 3>
- <cause 4 if applicable>

**IMPACT ON PERFORMANCE**
<describe how this defect affects energy output, reliability, or panel lifespan in 3-5 sentences>

**RECOMMENDED SOLUTION**
- <action step 1>
- <action step 2>
- <action step 3>
- <action step 4 if applicable>

**URGENCY LEVEL**
<one of: Low | Medium | High | Critical> — <brief reason>

Use enough detail to make this useful for maintenance planning, while keeping the language clear and practical. Provide full, helpful explanations rather than very short summaries.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def get_defect_explanation(defect_name: str, confidence: float = 0.0) -> dict:
    """
    Call Gemini and return a structured explanation for the detected defect.

    Parameters
    ----------
    defect_name : str    — top-1 class label from YOLO
    confidence  : float  — 0–100 (percentage); included in the prompt for context

    Returns
    -------
    {
      "success":     bool,
      "defect":      str,
      "confidence":  float,
      "explanation": str,   # raw Gemini markdown
      "sections": {
          "what_it_is": str,
          "causes":     str,
          "impact":     str,
          "solution":   str,
          "urgency":    str,
      },
      "error": str | None
    }
    """
    if not GEMINI_API_KEY:
        return _error_response(defect_name, confidence,
                               "GEMINI_API_KEY is not set in your .env file.")

    prompt = PROMPT_TEMPLATE.format(
        defect_name=defect_name.strip(),
        confidence=round(confidence, 1),
    )
    url = GEMINI_URL.format(api_key=GEMINI_API_KEY)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":    0.5,
            "topP":           0.95,
            "maxOutputTokens": 1024,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        raw_text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
        )

        if not raw_text:
            return _error_response(defect_name, confidence,
                                   "Gemini returned an empty response.")

        return {
            "success":     True,
            "defect":      defect_name,
            "confidence":  confidence,
            "explanation": raw_text,
            "sections":    _parse_sections(raw_text),
            "error":       None,
        }

    except requests.exceptions.Timeout:
        return _error_response(defect_name, confidence,
                               "Gemini API request timed out.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        msg    = _gemini_error_message(e.response)
        return _error_response(defect_name, confidence,
                               f"Gemini API error {status}: {msg}")
    except Exception as e:
        return _error_response(defect_name, confidence,
                               f"Unexpected error: {e}")


def is_api_key_configured() -> bool:
    """Return True when a non-empty API key is available."""
    return bool(GEMINI_API_KEY)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _error_response(defect_name: str, confidence: float, message: str) -> dict:
    return {
        "success":     False,
        "defect":      defect_name,
        "confidence":  confidence,
        "explanation": "",
        "sections":    {},
        "error":       message,
    }


def _parse_sections(text: str) -> dict:
    """Parse structured Gemini response; falls back gracefully."""
    mapping = {
        "what_it_is": r"\*\*WHAT IT IS\*\*\s*(.*?)(?=\*\*|$)",
        "causes":     r"\*\*CAUSES\*\*\s*(.*?)(?=\*\*|$)",
        "impact":     r"\*\*IMPACT ON PERFORMANCE\*\*\s*(.*?)(?=\*\*|$)",
        "solution":   r"\*\*RECOMMENDED SOLUTION\*\*\s*(.*?)(?=\*\*|$)",
        "urgency":    r"\*\*URGENCY LEVEL\*\*\s*(.*?)(?=\*\*|$)",
    }
    sections = {}
    for key, pattern in mapping.items():
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sections[key] = m.group(1).strip() if m else ""
    return sections


def _gemini_error_message(response) -> str:
    try:
        body = response.json()
        msg = body.get("error", {}).get("message", response.text[:200])
    except Exception:
        msg = response.text[:200] if response else "unknown"

    if response is not None:
        status = getattr(response, "status_code", None)
        if status == 429 or "quota" in msg.lower() or "rate limit" in msg.lower():
            return "Gemini quota exceeded. Please wait and try again later."

    return msg
