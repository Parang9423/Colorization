import json
import os
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def apply_gamma_float(gray_float: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    return np.power(np.clip(gray_float, 0.0, 1.0), gamma)


def apply_contrast(gray_float: np.ndarray, contrast: float) -> np.ndarray:
    # contrast=1.0: 원본 유지
    return np.clip((gray_float - 0.5) * contrast + 0.5, 0.0, 1.0)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_roi_bounds(img_shape, roi: Optional[Tuple[int, int, int, int]]):
    h_img, w_img = img_shape[:2]
    if roi is None:
        return 0, 0, w_img, h_img
    return roi


def grayscale_to_rgb_colorize(
    original_rgb: np.ndarray,
    base_rgb: Tuple[int, int, int],
    r_scale: float,
    g_scale: float,
    b_scale: float,
    gamma: float,
    contrast: float,
    intensity: float,
    saturation: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """흑백 luminance를 기반으로 3채널 RGB 이미지를 생성한다."""
    gray = rgb_to_gray(original_rgb).astype(np.float32) / 255.0
    gray = apply_contrast(gray, contrast)
    gray = apply_gamma_float(gray, gamma)

    base = np.array(base_rgb, dtype=np.float32) / 255.0
    channel_scale = np.array([r_scale, g_scale, b_scale], dtype=np.float32)
    color_vector = np.clip(base * channel_scale, 0.0, 1.0)

    colorized = gray[..., None] * color_vector[None, None, :] * intensity
    colorized = np.clip(colorized, 0.0, 1.0)

    # saturation 조절을 위해 HSV에서 S 채널만 배율 적용
    colorized_u8 = (colorized * 255).astype(np.uint8)
    hsv = cv2.cvtColor(colorized_u8, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch.astype(np.float32) * saturation, 0, 255).astype(np.uint8)
    colorized_u8 = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2RGB)

    result = original_rgb.copy()
    x, y, w, h = get_roi_bounds(original_rgb.shape, roi)
    result[y:y + h, x:x + w] = colorized_u8[y:y + h, x:x + w]
    return result


def adjust_hsv_region(
    img_rgb: np.ndarray,
    hue_shift: int,
    saturation_scale: float,
    value_scale: float,
    gamma: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    result = img_rgb.copy()
    x, y, w, h = get_roi_bounds(img_rgb.shape, roi)

    region = result[y:y + h, x:x + w]
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    h_ch = ((h_ch.astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    s_ch = np.clip(s_ch.astype(np.float32) * saturation_scale, 0, 255).astype(np.uint8)
    v_float = np.clip(v_ch.astype(np.float32) * value_scale / 255.0, 0, 1)
    v_ch = (apply_gamma_float(v_float, gamma) * 255).astype(np.uint8)

    adjusted_rgb = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2RGB)
    result[y:y + h, x:x + w] = adjusted_rgb
    return result


def save_result(img_rgb: np.ndarray, params: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(SAVE_DIR, f"colorized_{timestamp}.png")
    json_path = os.path.join(SAVE_DIR, f"params_{timestamp}.json")

    cv2.imwrite(image_path, rgb_to_bgr(img_rgb))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    return image_path, json_path


st.set_page_config(page_title="Colorization Palette Tool", layout="wide")
st.title("Colorization Palette & Gamma Tool")

uploaded_file = st.file_uploader(
    "이미지를 업로드하세요",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("흑백 이미지를 업로드하면 3채널 RGB 컬러 이미지 생성 UI가 표시됩니다.")
    st.stop()

pil_img = Image.open(uploaded_file)
original_rgb = pil_to_rgb(pil_img)
height, width = original_rgb.shape[:2]

st.sidebar.header("모드")
process_mode = st.sidebar.radio(
    "처리 방식",
    ["Grayscale → RGB Colorize", "RGB HSV Adjust"],
)

st.sidebar.header("적용 범위")
apply_mode = st.sidebar.radio("적용 범위", ["전체 이미지", "ROI 영역"])

roi = None
if apply_mode == "ROI 영역":
    st.sidebar.subheader("ROI 설정")
    x = st.sidebar.slider("X", 0, width - 1, 0)
    y = st.sidebar.slider("Y", 0, height - 1, 0)
    w = st.sidebar.slider("Width", 1, width - x, width - x)
    h = st.sidebar.slider("Height", 1, height - y, height - y)
    roi = (x, y, w, h)

if process_mode == "Grayscale → RGB Colorize":
    st.sidebar.header("RGB 컬러라이징")

    base_color = st.sidebar.color_picker("Base Color", "#C8A070")
    base_rgb = hex_to_rgb(base_color)

    r_scale = st.sidebar.slider("R Scale", 0.0, 3.0, 1.0, 0.05)
    g_scale = st.sidebar.slider("G Scale", 0.0, 3.0, 1.0, 0.05)
    b_scale = st.sidebar.slider("B Scale", 0.0, 3.0, 1.0, 0.05)

    intensity = st.sidebar.slider("Color Intensity", 0.0, 3.0, 1.0, 0.05)
    saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.05)
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.05)
    gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05)

    adjusted_rgb = grayscale_to_rgb_colorize(
        original_rgb=original_rgb,
        base_rgb=base_rgb,
        r_scale=r_scale,
        g_scale=g_scale,
        b_scale=b_scale,
        gamma=gamma,
        contrast=contrast,
        intensity=intensity,
        saturation=saturation,
        roi=roi,
    )

    params = {
        "source_file": uploaded_file.name,
        "process_mode": process_mode,
        "apply_mode": apply_mode,
        "roi": roi,
        "adjustment": {
            "base_color": base_color,
            "base_rgb": base_rgb,
            "r_scale": r_scale,
            "g_scale": g_scale,
            "b_scale": b_scale,
            "intensity": intensity,
            "saturation": saturation,
            "contrast": contrast,
            "gamma": gamma,
        },
    }
else:
    st.sidebar.header("HSV / Gamma 조절")

    hue_shift = st.sidebar.slider("Hue Shift", -90, 90, 0)
    saturation_scale = st.sidebar.slider("Saturation Scale", 0.0, 3.0, 1.0, 0.05)
    value_scale = st.sidebar.slider("Value Scale", 0.0, 3.0, 1.0, 0.05)
    gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05)

    adjusted_rgb = adjust_hsv_region(
        img_rgb=original_rgb,
        hue_shift=hue_shift,
        saturation_scale=saturation_scale,
        value_scale=value_scale,
        gamma=gamma,
        roi=roi,
    )

    params = {
        "source_file": uploaded_file.name,
        "process_mode": process_mode,
        "apply_mode": apply_mode,
        "roi": roi,
        "adjustment": {
            "hue_shift": hue_shift,
            "saturation_scale": saturation_scale,
            "value_scale": value_scale,
            "gamma": gamma,
        },
    }

left, center, right = st.columns(3)

with left:
    st.subheader("Original")
    st.image(original_rgb, use_container_width=True)

with center:
    st.subheader("Grayscale Input")
    gray_preview = rgb_to_gray(original_rgb)
    st.image(gray_preview, use_container_width=True, clamp=True)

with right:
    st.subheader("Adjusted RGB")
    st.image(adjusted_rgb, use_container_width=True)

st.subheader("보정 파라미터")
st.json(params)

if st.button("결과 저장"):
    image_path, json_path = save_result(adjusted_rgb, params)
    st.success("저장 완료")
    st.write(f"이미지 저장 경로: `{image_path}`")
    st.write(f"JSON 저장 경로: `{json_path}`")
