import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def load_images(uploaded_files) -> List[Dict]:
    images = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        lower_name = name.lower()

        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(IMAGE_EXTENSIONS) and not member.endswith("/"):
                        with zf.open(member) as fp:
                            images.append({
                                "name": os.path.basename(member),
                                "rgb": pil_to_rgb(Image.open(fp)),
                            })
        elif lower_name.endswith(IMAGE_EXTENSIONS):
            images.append({
                "name": name,
                "rgb": pil_to_rgb(Image.open(uploaded_file)),
            })

    return images


def image_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def make_zip(results: List[Dict], params: Dict) -> bytes:
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in results:
            stem = os.path.splitext(item["name"])[0]
            zf.writestr(f"colorized/{stem}_colorized.png", image_to_png_bytes(item["adjusted_rgb"]))
        zf.writestr("params.json", json.dumps(params, ensure_ascii=False, indent=2))

    buffer.seek(0)
    return buffer.getvalue()


def save_outputs_to_local(results: List[Dict], params: Dict, save_dir: str = "outputs") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"batch_{timestamp}")
    colorized_dir = os.path.join(run_dir, "colorized")
    os.makedirs(colorized_dir, exist_ok=True)

    for item in results:
        stem = os.path.splitext(item["name"])[0]
        cv2.imwrite(os.path.join(colorized_dir, f"{stem}_colorized.png"), rgb_to_bgr(item["adjusted_rgb"]))

    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    return run_dir
