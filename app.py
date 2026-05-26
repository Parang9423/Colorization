import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def resize_for_preview(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return img

    h, w = img.shape[:2]
    if w <= max_width:
        return img

    scale = max_width / w
    new_w = max_width
    new_h = max(1, int(h * scale))
    interpolation = cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def apply_gamma_float(gray_float: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    return np.power(np.clip(gray_float, 0.0, 1.0), gamma)


def apply_contrast(gray_float: np.ndarray, contrast: float) -> np.ndarray:
    return np.clip((gray_float - 0.5) * contrast + 0.5, 0.0, 1.0)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_roi_bounds(img_shape, roi: Optional[Tuple[int, int, int, int]]):
    h_img, w_img = img_shape[:2]
    if roi is None:
        return 0, 0, w_img, h_img

    x, y, w, h = roi
    x = int(np.clip(x, 0, w_img - 1))
    y = int(np.clip(y, 0, h_img - 1))
    w = int(np.clip(w, 1, w_img - x))
    h = int(np.clip(h, 1, h_img - y))
    return x, y, w, h


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
    gray = rgb_to_gray(original_rgb).astype(np.float32) / 255.0
    gray = apply_contrast(gray, contrast)
    gray = apply_gamma_float(gray, gamma)

    base = np.array(base_rgb, dtype=np.float32) / 255.0
    channel_scale = np.array([r_scale, g_scale, b_scale], dtype=np.float32)
    color_vector = np.clip(base * channel_scale, 0.0, 1.0)

    colorized = gray[..., None] * color_vector[None, None, :] * intensity
    colorized = np.clip(colorized, 0.0, 1.0)

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


def load_images_from_uploads(uploaded_files) -> List[Dict]:
    images = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        lower_name = name.lower()

        if lower_name.endswith(".zip"):
            zip_bytes = io.BytesIO(uploaded_file.getvalue())
            with zipfile.ZipFile(zip_bytes, "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(IMAGE_EXTENSIONS) and not member.endswith("/"):
                        with zf.open(member) as fp:
                            pil_img = Image.open(fp)
                            images.append({"name": os.path.basename(member), "rgb": pil_to_rgb(pil_img)})
        elif lower_name.endswith(IMAGE_EXTENSIONS):
            pil_img = Image.open(uploaded_file)
            images.append({"name": name, "rgb": pil_to_rgb(pil_img)})

    return images


def process_image(img_rgb: np.ndarray, params: dict, roi=None) -> np.ndarray:
    if params["process_mode"] == "Grayscale → RGB Colorize":
        return grayscale_to_rgb_colorize(
            original_rgb=img_rgb,
            base_rgb=params["base_rgb"],
            r_scale=params["r_scale"],
            g_scale=params["g_scale"],
            b_scale=params["b_scale"],
            gamma=params["gamma"],
            contrast=params["contrast"],
            intensity=params["intensity"],
            saturation=params["saturation"],
            roi=roi,
        )

    return adjust_hsv_region(
        img_rgb=img_rgb,
        hue_shift=params["hue_shift"],
        saturation_scale=params["saturation_scale"],
        value_scale=params["value_scale"],
        gamma=params["gamma"],
        roi=roi,
    )


def image_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def make_result_zip(results: List[Dict], params: dict) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in results:
            stem, _ = os.path.splitext(item["name"])
            zf.writestr(f"colorized/{stem}_colorized.png", image_to_png_bytes(item["adjusted_rgb"]))
        zf.writestr("params.json", json.dumps(params, ensure_ascii=False, indent=2))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def render_gallery(results: List[Dict], preview_count: int, columns_per_row: int, preview_width: int, layout_mode: str):
    visible_results = results[:preview_count]

    if layout_mode == "결과만 갤러리":
        for start in range(0, len(visible_results), columns_per_row):
            cols = st.columns(columns_per_row)
            for col, item in zip(cols, visible_results[start:start + columns_per_row]):
                with col:
                    st.caption(item["name"])
                    preview = resize_for_preview(item["adjusted_rgb"], preview_width)
                    st.image(preview, use_container_width=False)
        return

    if layout_mode == "원본/결과 2열 비교":
        for item in visible_results:
            st.markdown(f"#### {item['name']}")
            left, right = st.columns(2)
            with left:
                st.caption("Original")
                st.image(resize_for_preview(item["original_rgb"], preview_width), use_container_width=False)
            with right:
                st.caption("Adjusted RGB")
                st.image(resize_for_preview(item["adjusted_rgb"], preview_width), use_container_width=False)
        return

    for item in visible_results:
        st.markdown(f"#### {item['name']}")
        left, center, right = st.columns(3)
        with left:
            st.caption("Original")
            st.image(resize_for_preview(item["original_rgb"], preview_width), use_container_width=False)
        with center:
            st.caption("Grayscale")
            gray = rgb_to_gray(item["original_rgb"])
            st.image(resize_for_preview(gray, preview_width), use_container_width=False, clamp=True)
        with right:
            st.caption("Adjusted RGB")
            st.image(resize_for_preview(item["adjusted_rgb"], preview_width), use_container_width=False)


st.set_page_config(page_title="Colorization Palette Tool", layout="wide")
st.title("Colorization Palette & Gamma Tool")

uploaded_files = st.file_uploader(
    "이미지를 업로드하세요. 여러 이미지 또는 ZIP 업로드 가능",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("흑백 이미지 여러 장 또는 이미지가 들어있는 ZIP 파일을 업로드하세요.")
    st.stop()

images = load_images_from_uploads(uploaded_files)

if not images:
    st.warning("처리 가능한 이미지가 없습니다.")
    st.stop()

first_rgb = images[0]["rgb"]
height, width = first_rgb.shape[:2]

st.sidebar.header("모드")
process_mode = st.sidebar.radio("처리 방식", ["Grayscale → RGB Colorize", "RGB HSV Adjust"])

st.sidebar.header("적용 범위")
apply_mode = st.sidebar.radio("적용 범위", ["전체 이미지", "ROI 영역"])

roi = None
if apply_mode == "ROI 영역":
    st.sidebar.caption("ROI는 첫 번째 이미지 크기 기준으로 적용됩니다. 서로 크기가 다른 이미지는 범위가 자동 보정됩니다.")
    x = st.sidebar.slider("X", 0, width - 1, 0)
    y = st.sidebar.slider("Y", 0, height - 1, 0)
    w = st.sidebar.slider("Width", 1, width - x, width - x)
    h = st.sidebar.slider("Height", 1, height - y, height - y)
    roi = (x, y, w, h)

params = {"process_mode": process_mode, "apply_mode": apply_mode, "roi": roi}

if process_mode == "Grayscale → RGB Colorize":
    st.sidebar.header("RGB 컬러라이징")
    base_color = st.sidebar.color_picker("Base Color", "#C8A070")
    base_rgb = hex_to_rgb(base_color)

    params.update({
        "base_color": base_color,
        "base_rgb": base_rgb,
        "r_scale": st.sidebar.slider("R Scale", 0.0, 3.0, 1.0, 0.05),
        "g_scale": st.sidebar.slider("G Scale", 0.0, 3.0, 1.0, 0.05),
        "b_scale": st.sidebar.slider("B Scale", 0.0, 3.0, 1.0, 0.05),
        "intensity": st.sidebar.slider("Color Intensity", 0.0, 3.0, 1.0, 0.05),
        "saturation": st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.05),
        "contrast": st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.05),
        "gamma": st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05),
    })
else:
    st.sidebar.header("HSV / Gamma 조절")
    params.update({
        "hue_shift": st.sidebar.slider("Hue Shift", -90, 90, 0),
        "saturation_scale": st.sidebar.slider("Saturation Scale", 0.0, 3.0, 1.0, 0.05),
        "value_scale": st.sidebar.slider("Value Scale", 0.0, 3.0, 1.0, 0.05),
        "gamma": st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05),
    })

st.sidebar.header("Preview UI")
layout_mode = st.sidebar.radio(
    "Preview Layout",
    ["결과만 갤러리", "원본/결과 2열 비교", "원본/흑백/결과 3열 비교"],
)
columns_per_row = st.sidebar.slider("Columns per Row", 2, 8, 4)
preview_width = st.sidebar.slider("Preview Image Width(px)", 80, 500, 220, 20)
max_preview_limit = min(len(images), 100)
preview_count = st.sidebar.slider("Preview Count", 1, max_preview_limit, min(max_preview_limit, 12))

results = []
for item in images:
    adjusted = process_image(item["rgb"], params, roi=roi)
    results.append({"name": item["name"], "original_rgb": item["rgb"], "adjusted_rgb": adjusted})

st.subheader("Batch Preview")
st.write(f"처리 이미지 수: **{len(results)}** / 미리보기: **{preview_count}**")

render_gallery(
    results=results,
    preview_count=preview_count,
    columns_per_row=columns_per_row,
    preview_width=preview_width,
    layout_mode=layout_mode,
)

with st.expander("보정 파라미터 보기"):
    st.json(params)

zip_bytes = make_result_zip(results, params)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

st.download_button(
    label="결과 ZIP 다운로드",
    data=zip_bytes,
    file_name=f"colorized_results_{timestamp}.zip",
    mime="application/zip",
)

if st.button("로컬 outputs 폴더에 저장"):
    run_dir = os.path.join(SAVE_DIR, f"batch_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    colorized_dir = os.path.join(run_dir, "colorized")
    os.makedirs(colorized_dir, exist_ok=True)

    for item in results:
        stem, _ = os.path.splitext(item["name"])
        out_path = os.path.join(colorized_dir, f"{stem}_colorized.png")
        cv2.imwrite(out_path, rgb_to_bgr(item["adjusted_rgb"]))

    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    st.success(f"저장 완료: {run_dir}")
