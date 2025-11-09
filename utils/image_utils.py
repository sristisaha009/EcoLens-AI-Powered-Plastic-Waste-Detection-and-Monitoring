# utils/image_utils.py
import io, os, uuid
from PIL import Image, ExifTags
import numpy as np
import cv2
import exifread
from typing import Tuple, Optional

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """
    Streamlit UploadedFile -> saved file path (keeps extension)
    """
    fname = uploaded_file.name
    ext = os.path.splitext(fname)[1].lower()
    safe = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def load_image_cv2(path_or_file) -> np.ndarray:
    """
    Accepts:
      - path string
      - streamlit UploadedFile object
    Returns BGR image (as OpenCV convention) or raises.
    """
    if hasattr(path_or_file, "read"):
        # file-like
        data = path_or_file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode uploaded image")
        return img
    if isinstance(path_or_file, str):
        img = cv2.imread(path_or_file)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path_or_file}")
        return img
    raise TypeError("Unsupported input for load_image_cv2")

def pil_from_cv2(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def try_extract_exif_gps(filepath: str) -> Optional[Tuple[float,float]]:
    """
    Attempts to read EXIF GPS tags and return (lat, lon) in degrees.
    Returns None if not available.
    """
    try:
        with open(filepath, "rb") as f:
            tags = exifread.process_file(f, details=False)
        def _get(tag):
            return tags.get(tag)
        gps_lat = _get("GPS GPSLatitude")
        gps_lat_ref = _get("GPS GPSLatitudeRef")
        gps_lon = _get("GPS GPSLongitude")
        gps_lon_ref = _get("GPS GPSLongitudeRef")
        if not (gps_lat and gps_lon and gps_lat_ref and gps_lon_ref):
            return None
        def _to_deg(gps_ratio):
            nums = [float(x.num)/float(x.den) for x in gps_ratio.values]
            return nums[0] + nums[1]/60.0 + nums[2]/3600.0
        lat = _to_deg(gps_lat)
        if gps_lat_ref.values[0] != 'N':
            lat = -lat
        lon = _to_deg(gps_lon)
        if gps_lon_ref.values[0] != 'E':
            lon = -lon
        return (lat, lon)
    except Exception:
        return None
