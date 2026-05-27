import os
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st

from src.colorization import grayscale_to_rgb_colorize, rgb_to_gray
from src.dataset_qa import calculate_image_stats, detect_saturation_outliers, summarize_dataset
from src.evaluation import calculate_delta_e, calculate_psnr, calculate_rgb_histogram_similarity
from src.io_utils import load_images, make_zip, save_outputs_to_local

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def resize_keep_ratio(img: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return img
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def single_or_slider(label: str, min_value: int, max_value: int, default_value: int, key: str | None = None) -> int:
    if max_value <= min_value:
        st.caption(f"{label}: {min_value}")
        return min_value
    return st.slider(label, min_value, max_value, min(default_value, max_value), key=key)


def build_colorization_params() -> Dict:
    st.sidebar.header("Colorization Params")

    base_color = st.sidebar.color_picker("Base Color", "#C8A070")
    base_rgb = hex_to_rgb(base_color)

    params = {
        "mode": "Grayscale → RGB Colorize",
        "base_color": base_color,
        "base_rgb": base_rgb,
        "r_scale": st.sidebar.slider("R Scale", 0.0, 3.0, 1.0, 0.05),
        "g_scale": st.sidebar.slider("G Scale", 0.0, 3.0, 1.0, 0.05),
        "b_scale": st.sidebar.slider("B Scale", 0.0, 3.0, 1.0, 0.05),
        "intensity": st.sidebar.slider("Intensity", 0.0, 3.0, 1.0, 0.05),
        "saturation": st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.05),
        "contrast": st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.05),
        "gamma": st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.05),
    }

    return params


def process_images(images: List[Dict], params: Dict) -> List[Dict]:
    results = []

    for item in images:
        adjusted = grayscale_to_rgb_colorize(
            img_rgb=item["rgb"],
            base_rgb=params["base_rgb"],
            r_scale=params["r_scale"],
            g_scale=params["g_scale"],
            b_scale=params["b_scale"],
            gamma=params["gamma"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            intensity=params["intensity"],
        )

        results.append({
            "name": item["name"],
            "original_rgb": item["rgb"],
            "gray": rgb_to_gray(item["rgb"]),
            "adjusted_rgb": adjusted,
        })

    return results


def render_card(item: Dict, mode: str, preview_width: int):
    st.markdown(
        f"""
        <div style='border:1px solid #444;border-radius:10px;padding:8px;margin-bottom:8px;background:#1f1f1f;'>
            <div style='font-size:12px;font-weight:bold;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>
                {item['name']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "결과만 갤러리":
        st.image(resize_keep_ratio(item["adjusted_rgb"], preview_width), width=preview_width)
        return

    if mode == "원본/결과 2열 비교":
        image_width = max(40, preview_width // 2)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(resize_keep_ratio(item["original_rgb"], image_width), width=image_width)
        with c2:
            st.caption("Adjusted")
            st.image(resize_keep_ratio(item["adjusted_rgb"], image_width), width=image_width)
        return

    image_width = max(30, preview_width // 3)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Original")
        st.image(resize_keep_ratio(item["original_rgb"], image_width), width=image_width)
    with c2:
        st.caption("Gray")
        st.image(resize_keep_ratio(item["gray"], image_width), width=image_width, clamp=True)
    with c3:
        st.caption("Adjusted")
        st.image(resize_keep_ratio(item["adjusted_rgb"], image_width), width=image_width)


def render_gallery(results: List[Dict], mode: str, preview_count: int, columns_per_row: int, preview_width: int):
    visible = results[:preview_count]

    for start in range(0, len(visible), columns_per_row):
        row_items = visible[start:start + columns_per_row]
        cols = st.columns(columns_per_row)
        for col, item in zip(cols, row_items):
            with col:
                render_card(item, mode, preview_width)


def render_colorization_tab(results: List[Dict], params: Dict):
    st.subheader("Colorization")
    st.write(f"처리 이미지 수: **{len(results)}**")

    with st.expander("Preview UI 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            preview_mode = st.selectbox(
                "Preview Layout",
                ["결과만 갤러리", "원본/결과 2열 비교", "원본/흑백/결과 3열 비교"],
                index=0,
            )
        with c2:
            columns_per_row = st.slider("Columns per Row", 1, 8, 4)
        with c3:
            preview_width = st.slider("Preview Card Width(px)", 80, 900, 220, 20)
        with c4:
            preview_count = single_or_slider(
                "Preview Count",
                1,
                len(results),
                min(len(results), 12),
                key="color_preview_count",
            )

    render_gallery(results, preview_mode, preview_count, columns_per_row, preview_width)

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


def match_gt_images(results: List[Dict], gt_images: List[Dict]) -> List[Dict]:
    gt_map = {item["name"]: item["rgb"] for item in gt_images}
    matched = []

    for result in results:
        if result["name"] in gt_map:
            gt = gt_map[result["name"]]
            pred = result["adjusted_rgb"]
            if gt.shape[:2] != pred.shape[:2]:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
            matched.append({**result, "gt_rgb": gt})

    return matched


def render_evaluation_tab(results: List[Dict]):
    st.subheader("Evaluation")
    st.write("실사 GT 이미지와 현재 컬러라이징 결과를 비교합니다. 파일명 기준으로 매칭합니다.")

    gt_files = st.file_uploader(
        "GT 컬러 이미지 업로드 또는 ZIP 업로드",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
        accept_multiple_files=True,
        key="gt_uploader",
    )

    if not gt_files:
        st.info("GT 이미지를 업로드하면 RGB Histogram Similarity, LAB Delta E, PSNR을 계산합니다.")
        return

    gt_images = load_images(gt_files)
    matched = match_gt_images(results, gt_images)

    if not matched:
        st.warning("파일명이 일치하는 GT 이미지가 없습니다. 예: sample_001.png ↔ sample_001.png")
        return

    rows = []
    for item in matched:
        hist = calculate_rgb_histogram_similarity(item["gt_rgb"], item["adjusted_rgb"])
        delta = calculate_delta_e(item["gt_rgb"], item["adjusted_rgb"])
        psnr = calculate_psnr(item["gt_rgb"], item["adjusted_rgb"])

        rows.append({
            "name": item["name"],
            "hist_mean": round(hist["mean"], 4),
            "delta_e_mean": round(delta["mean_delta_e"], 4),
            "delta_e_max": round(delta["max_delta_e"], 4),
            "psnr": round(psnr, 4),
        })

    st.write(f"매칭된 이미지 수: **{len(matched)}**")
    st.table(rows)

    st.markdown("### Metric Guide")
    st.write("- Histogram Similarity: 1에 가까울수록 RGB 분포가 유사")
    st.write("- Delta E: 낮을수록 LAB 시각 색상 거리 작음")
    st.write("- PSNR: 높을수록 픽셀 단위 차이가 작음")

    preview_count = single_or_slider(
        "Evaluation Preview Count",
        1,
        len(matched),
        min(len(matched), 6),
        key="eval_preview_count",
    )
    for item in matched[:preview_count]:
        st.markdown(f"#### {item['name']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("GT")
            st.image(item["gt_rgb"], use_container_width=True)
        with c2:
            st.caption("Pred")
            st.image(item["adjusted_rgb"], use_container_width=True)
        with c3:
            st.caption("Gray")
            st.image(item["gray"], use_container_width=True, clamp=True)


def render_dataset_qa_tab(results: List[Dict]):
    st.subheader("Dataset QA")
    st.write("현재 컬러라이징 결과 데이터셋의 색상 통계와 이상치를 확인합니다.")

    summary = summarize_dataset(results)
    if not summary:
        st.warning("QA를 계산할 이미지가 없습니다.")
        return

    st.markdown("### Dataset Summary")
    summary_rows = []
    for key, values in summary.items():
        summary_rows.append({
            "metric": key,
            "mean": round(values["mean"], 4),
            "std": round(values["std"], 4),
            "min": round(values["min"], 4),
            "max": round(values["max"], 4),
        })
    st.table(summary_rows)

    st.markdown("### Per-image Stats")
    rows = []
    for item in results:
        stats = calculate_image_stats(item["adjusted_rgb"])
        rows.append({"name": item["name"], **{k: round(v, 4) for k, v in stats.items()}})
    st.table(rows)

    st.markdown("### Saturation Outlier Detection")
    z_threshold = st.slider("Saturation Z-score Threshold", 1.0, 4.0, 2.0, 0.1)
    outliers = detect_saturation_outliers(results, z_threshold=z_threshold)

    if outliers:
        st.warning(f"Saturation outlier {len(outliers)}개 감지")
        st.table(outliers)
    else:
        st.success("Saturation 기준 이상치가 없습니다.")


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

params = build_colorization_params()
results = process_images(images, params)

tab_color, tab_eval, tab_qa = st.tabs(["Colorization", "Evaluation", "Dataset QA"])

with tab_color:
    render_colorization_tab(results, params)

with tab_eval:
    render_evaluation_tab(results)

with tab_qa:
    render_dataset_qa_tab(results)
