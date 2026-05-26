import json
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_gamma(channel: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    normalized = channel.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def adjust_hsv_region(
    img_bgr: np.ndarray,
    hue_shift: int,
    saturation_scale: float,
    value_scale: float,
    gamma: float,
    roi=None,
) -> np.ndarray:
    result = img_bgr.copy()

    if roi is None:
        x, y, w, h = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
    else:
        x, y, w, h = roi

    region = result[y:y + h, x:x + w]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    h_ch, s_ch, v_ch = cv2.split(hsv)

    h_ch = ((h_ch.astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    s_ch = np.clip(s_ch.astype(np.float32) * saturation_scale, 0, 255).astype(np.uint8)
    v_ch = np.clip(v_ch.astype(np.float32) * value_scale, 0, 255).astype(np.uint8)
    v_ch = apply_gamma(v_ch, gamma)

    adjusted_hsv = cv2.merge([h_ch, s_ch, v_ch])
    adjusted_bgr = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    result[y:y + h, x:x + w] = adjusted_bgr

    return result


st.set_page_config(
    page_title="Colorization Palette Tool",
    layout="wide"
)

st.title("Colorization Palette & Gamma Tool")

uploaded_file = st.file_uploader(
    "이미지를 업로드하세요",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is None:
    st.info("흑백 또는 컬러 이미지를 업로드하면 보정 UI가 표시됩니다.")
    st.stop()

pil_img = Image.open(uploaded_file)
original_bgr = pil_to_cv2(pil_img)

height, width = original_bgr.shape[:2]

st.sidebar.header("보정 범위")

mode = st.sidebar.radio(
    "적용 범위",
    ["전체 이미지", "ROI 영역"]
)

roi = None

if mode == "ROI 영역":
    st.sidebar.subheader("ROI 설정")

    x = st.sidebar.slider("X", 0, width - 1, 0)
    y = st.sidebar.slider("Y", 0, height - 1, 0)

    max_w = width - x
    max_h = height - y

    w = st.sidebar.slider("Width", 1, max_w, max_w)
    h = st.sidebar.slider("Height", 1, max_h, max_h)

    roi = (x, y, w, h)

st.sidebar.header("HSV / Gamma 조절")

hue_shift = st.sidebar.slider(
    "Hue Shift",
    min_value=-90,
    max_value=90,
    value=0,
)

saturation_scale = st.sidebar.slider(
    "Saturation Scale",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

value_scale = st.sidebar.slider(
    "Value Scale",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

gamma = st.sidebar.slider(
    "Gamma",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

adjusted_bgr = adjust_hsv_region(
    img_bgr=original_bgr,
    hue_shift=hue_shift,
    saturation_scale=saturation_scale,
    value_scale=value_scale,
    gamma=gamma,
    roi=roi
)

left, right = st.columns(2)

with left:
    st.subheader("Original")
    st.image(cv2_to_rgb(original_bgr), use_container_width=True)

with right:
    st.subheader("Adjusted")
    st.image(cv2_to_rgb(adjusted_bgr), use_container_width=True)

params = {
    "source_file": uploaded_file.name,
    "mode": mode,
    "roi": roi,
    "adjustment": {
        "hue_shift": hue_shift,
        "saturation_scale": saturation_scale,
        "value_scale": value_scale,
        "gamma": gamma,
    }
}

st.json(params)

if st.button("결과 저장"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(SAVE_DIR, f"colorized_{timestamp}.png")
    json_path = os.path.join(SAVE_DIR, f"params_{timestamp}.json")

    cv2.imwrite(image_path, adjusted_bgr)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    st.success("저장 완료")
