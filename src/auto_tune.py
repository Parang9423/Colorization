from typing import Dict, List

import cv2
import numpy as np


def _safe_ratio(gt_value: float, pred_value: float, min_ratio: float = 0.5, max_ratio: float = 1.5) -> float:
    pred_value = max(float(pred_value), 1e-6)
    ratio = float(gt_value) / pred_value
    return float(np.clip(ratio, min_ratio, max_ratio))


def _rgb_to_hex(rgb_values: np.ndarray) -> str:
    rgb = np.clip(np.round(rgb_values), 0, 255).astype(np.uint8)
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _channel_direction(ratio: float, threshold: float = 0.03) -> str:
    if ratio > 1.0 + threshold:
        return "increase"
    if ratio < 1.0 - threshold:
        return "decrease"
    return "keep"


def _direction_label(direction: str) -> str:
    if direction == "increase":
        return "증가 권장"
    if direction == "decrease":
        return "감소 권장"
    return "유지 권장"


def recommend_rgb_balance(matched_items: List[Dict], current_params: Dict) -> Dict:
    """Recommend RGB scale values by comparing matched GT and prediction images.

    The recommendation uses dataset-level channel mean ratios:
    recommended_scale = current_scale * mean(GT_channel / Pred_channel)

    This is intentionally simple and explainable for operator-facing QA tools.
    """
    if not matched_items:
        return {
            "available": False,
            "message": "GT와 매칭된 이미지가 없어 RGB Scale 추천을 생성할 수 없습니다.",
        }

    gt_means = []
    pred_means = []

    for item in matched_items:
        gt = item["gt_rgb"]
        pred = item["adjusted_rgb"]

        if gt.shape[:2] != pred.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        gt_means.append(np.mean(gt.reshape(-1, 3), axis=0))
        pred_means.append(np.mean(pred.reshape(-1, 3), axis=0))

    gt_mean = np.mean(np.stack(gt_means, axis=0), axis=0)
    pred_mean = np.mean(np.stack(pred_means, axis=0), axis=0)

    ratios = np.array([
        _safe_ratio(gt_mean[0], pred_mean[0]),
        _safe_ratio(gt_mean[1], pred_mean[1]),
        _safe_ratio(gt_mean[2], pred_mean[2]),
    ], dtype=np.float32)

    current_scales = np.array([
        float(current_params.get("r_scale", 1.0)),
        float(current_params.get("g_scale", 1.0)),
        float(current_params.get("b_scale", 1.0)),
    ], dtype=np.float32)

    recommended_scales = np.clip(current_scales * ratios, 0.0, 3.0)

    gt_warmth = float(gt_mean[0] - gt_mean[2])
    pred_warmth = float(pred_mean[0] - pred_mean[2])
    warmth_gap = pred_warmth - gt_warmth

    if warmth_gap > 8:
        warmth_bias = "Pred 결과가 GT보다 따뜻한 톤입니다. R Scale 감소 또는 B Scale 증가를 우선 검토하세요."
    elif warmth_gap < -8:
        warmth_bias = "Pred 결과가 GT보다 차가운 톤입니다. B Scale 감소 또는 R Scale 증가를 우선 검토하세요."
    else:
        warmth_bias = "Warm/Cool balance는 큰 편차가 없습니다. 채널별 미세 조정 위주로 확인하세요."

    channel_names = ["R", "G", "B"]
    channel_details = []

    for idx, channel in enumerate(channel_names):
        direction = _channel_direction(float(ratios[idx]))
        channel_details.append({
            "channel": channel,
            "gt_mean": round(float(gt_mean[idx]), 3),
            "pred_mean": round(float(pred_mean[idx]), 3),
            "ratio": round(float(ratios[idx]), 4),
            "current_scale": round(float(current_scales[idx]), 4),
            "recommended_scale": round(float(recommended_scales[idx]), 4),
            "direction": direction,
            "direction_label": _direction_label(direction),
        })

    confidence = 1.0 / (1.0 + float(np.mean(np.abs(ratios - 1.0))))

    return {
        "available": True,
        "gt_mean_rgb": [round(float(v), 3) for v in gt_mean],
        "pred_mean_rgb": [round(float(v), 3) for v in pred_mean],
        "recommended_base_color": _rgb_to_hex(gt_mean),
        "recommended_scales": {
            "r_scale": round(float(recommended_scales[0]), 4),
            "g_scale": round(float(recommended_scales[1]), 4),
            "b_scale": round(float(recommended_scales[2]), 4),
        },
        "channel_details": channel_details,
        "warmth_bias": warmth_bias,
        "confidence": round(float(confidence), 4),
        "note": "추천값은 GT/Pred 평균 RGB 비율 기반입니다. 자동 적용이 아니라 튜닝 방향 제안용으로 사용하세요.",
    }
