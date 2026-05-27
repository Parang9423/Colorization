import cv2
import numpy as np


def calculate_rgb_histogram_similarity(img1: np.ndarray, img2: np.ndarray):
    scores = {}

    for idx, channel in enumerate(["R", "G", "B"]):
        hist1 = cv2.calcHist([img1], [idx], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [idx], None, [256], [0, 256])

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        scores[channel] = float(score)

    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def calculate_delta_e(img1: np.ndarray, img2: np.ndarray):
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)

    delta = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    return {
        "mean_delta_e": float(np.mean(delta)),
        "max_delta_e": float(np.max(delta)),
        "min_delta_e": float(np.min(delta)),
    }


def calculate_psnr(img1: np.ndarray, img2: np.ndarray):
    psnr = cv2.PSNR(img1, img2)
    return float(psnr)
