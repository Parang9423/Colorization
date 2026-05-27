import cv2
import numpy as np
from typing import Tuple


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    return np.power(np.clip(gray, 0.0, 1.0), gamma)


def apply_contrast(gray: np.ndarray, contrast: float) -> np.ndarray:
    return np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)


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
