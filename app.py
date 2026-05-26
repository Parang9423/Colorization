import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def resize_keep_ratio(img: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return img

    h, w = img.shape[:2]
    if w <= width:
        return img

    scale = width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    return np.power(np.clip(gray, 0.0, 1.0), gamma)


def apply_contrast(gray: np.ndarray, contrast: float) -> np.ndarray:
    return np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def grayscale_to_rgb_colorize(
    img_rgb: np.ndarray,
    base_rgb: Tuple[int, int, int],
    r_scale: float,
    g_scale: float,
    b_scale: float,
    gamma: float,
    contrast: float,
    saturation: float,
    intensity: float,
) -> np.ndarray:
    gray = rgb_to_gray(img_rgb).astype(np.float32) / 255.0
    gray = apply_contrast(gray, contrast)
    gray = apply_gamma(gray, gamma)

    base = np.array(base_rgb, dtype=np.float32) / 255.0
    scales = np.array([r_scale, g_scale, b_scale], dtype=np.float32)
    palette = np.clip(base * scales, 0.0, 1.0)

    colorized = gray[..., None] * palette[None, None, :] * intensity
    colorized = np.clip(colorized, 0.0, 1.0)
    colorized_u8 = (colorized * 255).astype(np.uint8)

    hsv = cv2.cvtColor(colorized_u8, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch.astype(np.float32) * saturation, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2RGB)


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
                            pil_img = Image.open(fp)
                            images.append({
                                "name": os.path.basename(member),
                                "rgb": pil_to_rgb(pil_img),
                            })
        elif lower_name.endswith(IMAGE_EXTENSIONS):
            pil_img = Image.open(uploaded_file)
            images.append({
                "name": name,
                "rgb": pil_to_rgb(pil_img),
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
            zf.writestr(
                f"colorized/{stem}_colorized.png",
                image_to_png_bytes(item["adjusted_rgb"]),
            )

        zf.writestr(
            "params.json",
            json.dumps(params, ensure_ascii=False, indent=2),
        )

    buffer.seek(0)
    return buffer.getvalue()


def save_outputs_to_local(results: List[Dict], params: Dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"batch_{timestamp}")
    colorized_dir = os.path.join(run_dir, "colorized")
    os.makedirs(colorized_dir, exist_ok=True)

    for item in results:
        stem = os.path.splitext(item["name"])[0]
        out_path = os.path.join(colorized_dir, f"{stem}_colorized.png")
        cv2.imwrite(out_path, rgb_to_bgr(item["adjusted_rgb"]))

    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    return run_dir


def render_card(item: Dict, mode: str, preview_width: int):
    st.markdown(
        f"""
        <div style='
            border:1px solid #444;
            border-radius:10px;
            padding:8px;
            margin-bottom:8px;
            background:#1f1f1f;
        '>
            <div style='
                font-size:12px;
                font-weight:bold;
                overflow:hidden;
                text-overflow:ellipsis;
                white-space:nowrap;
            '>{item['name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "결과만 갤러리":
        st.image(
            resize_keep_ratio(item["adjusted_rgb"], preview_width),
            width=preview_width,
        )
        return

    if mode == "원본/결과 2열 비교":
        image_width = max(40, preview_width // 2)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(
                resize_keep_ratio(item["original_rgb"], image_width),
                width=image_width,
            )
        with c2:
            st.caption("Adjusted")
            st.image(
                resize_keep_ratio(item["adjusted_rgb"], image_width),
                width=image_width,
            )
        return

    image_width = max(30, preview_width // 3)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Original")
        st.image(
            resize_keep_ratio(item["original_rgb"], image_width),
            width=image_width,
        )
    with c2:
        st.caption("Gray")
        st.image(
            resize_keep_ratio(rgb_to_gray(item["original_rgb"]), image_width),
            width=image_width,
            clamp=True,
        )
    with c3:
        st.caption("Adjusted")
        st.image(
            resize_keep_ratio(item["adjusted_rgb"], image_width),
            width=image_width,
        )


def render_gallery(
    results: List[Dict],
    mode: str,
    preview_count: int,
    columns_per_row: int,
    preview_width: int,
):
    visible = results[:preview_count]

    for start in range(0, len(visible), columns_per_row):
        row_items = visible[start:start + columns_per_row]
        cols = st.columns(columns_per_row)

        for col, item in zip(cols, row_items):
            with col:
                render_card(item, mode, preview_width)


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

images = load_images(uploaded_files)

if not images:
    st.warning("처리 가능한 이미지가 없습니다.")
    st.stop()

st.sidebar.header("Colorization")
base_color = st.sidebar.color_picker("Base Color", "#C8A070")
base_rgb = hex_to_rgb(base_color)

r_scale = st.sidebar.slider("R Scale", 0.0, 3.0, 1.0, 0.05)
g_scale = st.sidebar.slider("G Scale", 0.0, 3.0, 1.0, 0.05)
b_scale = st.sidebar.slider("B Scale", 0.0, 3.0, 1.0, 0.05)
intensity = st.sidebar.slider("Intensity", 0.0, 3.0, 1.0, 0.05)
saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.05)
contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.05)
gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05)

params = {
    "mode": "Grayscale → RGB Colorize",
    "base_color": base_color,
    "base_rgb": base_rgb,
    "r_scale": r_scale,
    "g_scale": g_scale,
    "b_scale": b_scale,
    "intensity": intensity,
    "saturation": saturation,
    "contrast": contrast,
    "gamma": gamma,
}

results = []
for item in images:
    adjusted = grayscale_to_rgb_colorize(
        img_rgb=item["rgb"],
        base_rgb=base_rgb,
        r_scale=r_scale,
        g_scale=g_scale,
        b_scale=b_scale,
        gamma=gamma,
        contrast=contrast,
        saturation=saturation,
        intensity=intensity,
    )

    results.append({
        "name": item["name"],
        "original_rgb": item["rgb"],
        "adjusted_rgb": adjusted,
    })

st.subheader("Batch Preview")
st.write(f"처리 이미지 수: **{len(results)}**")

with st.expander("Preview UI 설정", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        preview_mode = st.selectbox(
            "Preview Layout",
            [
                "결과만 갤러리",
                "원본/결과 2열 비교",
                "원본/흑백/결과 3열 비교",
            ],
            index=0,
        )

    with c2:
        columns_per_row = st.slider("Columns per Row", 1, 8, 4)

    with c3:
        preview_width = st.slider("Preview Card Width(px)", 80, 900, 220, 20)

    with c4:
        preview_count = st.slider(
            "Preview Count",
            1,
            len(results),
            min(len(results), 12),
        )

st.write(
    f"Preview: **{preview_count}** / Layout: **{preview_mode}** / Card Width: **{preview_width}px**"
)

render_gallery(
    results=results,
    mode=preview_mode,
    preview_count=preview_count,
    columns_per_row=columns_per_row,
    preview_width=preview_width,
)

with st.expander("파라미터 보기"):
    st.json(params)

zip_bytes = make_zip(results, params)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

st.download_button(
    "결과 ZIP 다운로드",
    data=zip_bytes,
    file_name=f"colorized_{timestamp}.zip",
    mime="application/zip",
)

if st.button("로컬 outputs 폴더에 저장"):
    run_dir = save_outputs_to_local(results, params)
    st.success(f"저장 완료: {run_dir}")
