import cv2
import numpy as np
from typing import Dict, List


def calculate_image_stats(img_rgb: np.ndarray) -> Dict:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    rgb_mean = np.mean(img_rgb, axis=(0, 1))
    rgb_std = np.std(img_rgb, axis=(0, 1))

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    l_mean = float(np.mean(lab[:, :, 0]))
    a_mean = float(np.mean(lab[:, :, 1]))
    b_mean = float(np.mean(lab[:, :, 2]))

    return {
        "rgb_mean_r": float(rgb_mean[0]),
        "rgb_mean_g": float(rgb_mean[1]),
        "rgb_mean_b": float(rgb_mean[2]),
        "rgb_std_r": float(rgb_std[0]),
        "rgb_std_g": float(rgb_std[1]),
        "rgb_std_b": float(rgb_std[2]),
        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,
        "l_mean": l_mean,
        "a_mean": a_mean,
        "b_mean": b_mean,
    }


def summarize_dataset(results: List[Dict]) -> Dict:
    if not results:
        return {}

    stats = [calculate_image_stats(item["adjusted_rgb"]) for item in results]
    keys = stats[0].keys()

    summary = {}
    for key in keys:
        values = np.array([item[key] for item in stats], dtype=np.float32)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return summary


def detect_saturation_outliers(results: List[Dict], z_threshold: float = 2.0) -> List[Dict]:
    if len(results) < 2:
        return []

    records = []
    for item in results:
        stats = calculate_image_stats(item["adjusted_rgb"])
        records.append({"name": item["name"], "s_mean": stats["s_mean"]})

    values = np.array([item["s_mean"] for item in records], dtype=np.float32)
    mean = np.mean(values)
    std = np.std(values) + 1e-6

    outliers = []
    for record in records:
        z = float((record["s_mean"] - mean) / std)
        if abs(z) >= z_threshold:
            outliers.append({
                "name": record["name"],
                "s_mean": float(record["s_mean"]),
                "z_score": z,
            })

    return outliers
