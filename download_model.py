"""
download_model.py  —  Google Drive downloader for Render deployment

If `model/best.pt` is not present, this helper will download it from a
Google Drive share link at startup so Render can launch without a prebuilt
model bundle inside the repository.
"""

import os
import re
from typing import Optional

import requests

GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/file/d/1edJY_qKs3d1h9XgQ7Z9a9J5tF1kVIbSm/view?usp=sharing"
CHUNK_SIZE = 32 * 1024


def parse_google_drive_file_id(url: str) -> Optional[str]:
    """Extract a Google Drive file ID from a shared link."""
    if not url:
        return None

    patterns = [
        r"drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/uc\?export=download&id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Sometimes the user may pass the raw file ID directly.
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,100}", url):
        return url

    return None


def ensure_model_downloaded(destination: str, download_url: str, timeout: int = 120) -> bool:
    """Download the model file if it is missing, returning True on success."""
    if os.path.exists(destination):
        return True

    if not download_url:
        return False

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    file_id = parse_google_drive_file_id(download_url)

    if file_id:
        return _download_file_from_google_drive(file_id, destination, timeout)

    return _download_file_direct(download_url, destination, timeout)


def _download_file_from_google_drive(file_id: str, destination: str, timeout: int) -> bool:
    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_DOWNLOAD_URL, params={"id": file_id}, stream=True, timeout=timeout)

    token = _get_confirm_token(response)
    if token:
        response = session.get(
            GOOGLE_DRIVE_DOWNLOAD_URL,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=timeout,
        )

    _save_response_content(response, destination)
    return True


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    match = re.search(r"confirm=([0-9A-Za-z_\-]+)&", response.text)
    return match.group(1) if match else None


def _download_file_direct(url: str, destination: str, timeout: int) -> bool:
    response = requests.get(url, stream=True, timeout=timeout)
    _save_response_content(response, destination)
    return True


def _save_response_content(response: requests.Response, destination: str) -> None:
    if response.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type and "download" not in response.headers.get("content-disposition", ""):
        raise RuntimeError("Download failed: Google Drive returned an HTML page instead of a file.")

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    if os.path.getsize(destination) == 0:
        raise RuntimeError("Download failed: destination file is empty.")
