from typing import Dict, List, Optional


def _advice(level: str, title: str, reason: str, actions: List[str]) -> Dict:
    return {
        "level": level,
        "title": title,
        "reason": reason,
        "actions": actions,
    }


def generate_evaluation_advice(rows: List[Dict]) -> List[Dict]:
    """Generate rule-based recommendations from Evaluation metrics."""
    if not rows:
        return [
            _advice(
                "INFO",
                "평가 데이터 부족",
                "GT와 매칭된 이미지가 없어 평가 기반 추천을 생성할 수 없습니다.",
                ["GT 파일명과 입력 이미지 파일명이 동일한지 확인하세요."],
            )
        ]

    hist_mean = sum(row["hist_mean"] for row in rows) / len(rows)
    delta_e_mean = sum(row["delta_e_mean"] for row in rows) / len(rows)
    psnr_mean = sum(row["psnr"] for row in rows) / len(rows)

    advice = []

    if delta_e_mean > 20:
        advice.append(
            _advice(
                "CRITICAL",
                "GT 대비 색상 차이가 매우 큼",
                f"평균 Delta E가 {delta_e_mean:.2f}로 높습니다. 사람 눈 기준으로 색상 차이가 명확할 가능성이 큽니다.",
                [
                    "Base Color를 GT의 대표 색상에 더 가깝게 변경하세요.",
                    "R/G/B Scale을 한 번에 크게 바꾸기보다 한 채널씩 0.05~0.15 단위로 조정하세요.",
                    "색이 과하게 따뜻하면 R Scale을 낮추고 B Scale을 올리세요.",
                    "색이 차갑거나 푸르게 보이면 B Scale을 낮추고 R Scale을 올리세요.",
                ],
            )
        )
    elif delta_e_mean > 10:
        advice.append(
            _advice(
                "WARNING",
                "색상 차이 조정 필요",
                f"평균 Delta E가 {delta_e_mean:.2f}입니다. 전체 색감은 유사해도 세부 색상 차이가 보일 수 있습니다.",
                [
                    "Base Color와 RGB Scale을 미세 조정하세요.",
                    "Saturation을 0.05~0.10 단위로 낮춰 과채색 여부를 확인하세요.",
                    "GT/Pred 비교 이미지를 보며 특정 색상 bias가 있는지 확인하세요.",
                ],
            )
        )
    else:
        advice.append(
            _advice(
                "GOOD",
                "색상 거리 양호",
                f"평균 Delta E가 {delta_e_mean:.2f}로 비교적 안정적입니다.",
                ["현재 RGB balance를 유지하면서 Histogram과 PSNR을 기준으로 미세 조정하세요."],
            )
        )

    if hist_mean < 0.5:
        advice.append(
            _advice(
                "CRITICAL",
                "전체 RGB 분포 차이가 큼",
                f"평균 Histogram Similarity가 {hist_mean:.2f}로 낮습니다. 전체 색감 분위기가 GT와 다를 가능성이 큽니다.",
                [
                    "Base Color를 재설정하세요.",
                    "R/G/B Scale 중 한 채널이 과하게 높거나 낮지 않은지 확인하세요.",
                    "Saturation을 낮춰 색 분포가 과도하게 벌어지는지 확인하세요.",
                ],
            )
        )
    elif hist_mean < 0.8:
        advice.append(
            _advice(
                "WARNING",
                "RGB 분포 유사도 개선 필요",
                f"평균 Histogram Similarity가 {hist_mean:.2f}입니다. 전체 색감 분포가 완전히 맞지는 않습니다.",
                [
                    "Saturation과 Intensity를 0.05~0.10 단위로 조정하세요.",
                    "GT가 저채도 이미지라면 Saturation을 낮추는 방향을 우선 테스트하세요.",
                    "GT가 밝은 이미지라면 Intensity 또는 Gamma를 조정하세요.",
                ],
            )
        )
    else:
        advice.append(
            _advice(
                "GOOD",
                "RGB 분포 유사도 양호",
                f"평균 Histogram Similarity가 {hist_mean:.2f}로 비교적 높습니다.",
                ["전체 색감 분포는 유지하고 Delta E가 높은 개별 이미지 위주로 확인하세요."],
            )
        )

    if psnr_mean < 15:
        advice.append(
            _advice(
                "WARNING",
                "픽셀 단위 차이가 큼",
                f"평균 PSNR이 {psnr_mean:.2f}로 낮습니다. 명암, contrast, intensity 차이가 클 수 있습니다.",
                [
                    "Contrast를 0.9~1.0 범위로 낮춰보세요.",
                    "Gamma를 0.9~1.1 범위에서 조정해 밝기 왜곡을 줄이세요.",
                    "Intensity가 너무 높으면 clipping이 발생할 수 있으므로 낮춰보세요.",
                ],
            )
        )
    elif psnr_mean < 25:
        advice.append(
            _advice(
                "INFO",
                "명암 차이 점검 필요",
                f"평균 PSNR이 {psnr_mean:.2f}입니다. 구조는 유지되지만 명암 차이가 존재할 수 있습니다.",
                [
                    "Gamma와 Contrast를 우선 미세 조정하세요.",
                    "색감보다 밝기 차이가 커 보이면 RGB Scale보다 Gamma를 먼저 조정하세요.",
                ],
            )
        )

    return advice


def generate_dataset_qa_advice(summary: Dict, outliers: Optional[List[Dict]] = None) -> List[Dict]:
    """Generate rule-based recommendations from dataset-level QA stats."""
    if not summary:
        return [
            _advice(
                "INFO",
                "QA 데이터 부족",
                "Dataset Summary가 없어 추천을 생성할 수 없습니다.",
                ["이미지를 먼저 업로드하고 컬러라이징 결과를 생성하세요."],
            )
        ]

    outliers = outliers or []
    advice = []

    s_mean = summary.get("s_mean", {}).get("mean")
    s_std = summary.get("s_mean", {}).get("std")
    v_mean = summary.get("v_mean", {}).get("mean")
    rgb_std_r = summary.get("rgb_mean_r", {}).get("std")
    rgb_std_g = summary.get("rgb_mean_g", {}).get("std")
    rgb_std_b = summary.get("rgb_mean_b", {}).get("std")

    if s_mean is not None:
        if s_mean > 120:
            advice.append(
                _advice(
                    "WARNING",
                    "데이터셋 평균 채도가 높음",
                    f"평균 Saturation이 {s_mean:.2f}입니다. 실제 저채도/금속성 이미지라면 과채색 가능성이 있습니다.",
                    [
                        "Saturation을 0.7~0.9 범위로 낮춰 테스트하세요.",
                        "Intensity가 높다면 함께 낮춰 clipping을 방지하세요.",
                    ],
                )
            )
        elif s_mean < 30:
            advice.append(
                _advice(
                    "INFO",
                    "데이터셋 평균 채도가 낮음",
                    f"평균 Saturation이 {s_mean:.2f}입니다. 결과가 지나치게 회색/무채색에 가까울 수 있습니다.",
                    [
                        "Saturation을 1.05~1.20 범위로 조금 올려보세요.",
                        "Base Color가 지나치게 중성 회색인지 확인하세요.",
                    ],
                )
            )

    if s_std is not None and s_std > 35:
        advice.append(
            _advice(
                "WARNING",
                "이미지 간 채도 편차가 큼",
                f"Saturation 표준편차가 {s_std:.2f}입니다. 이미지별 색감 consistency가 낮을 수 있습니다.",
                [
                    "Saturation 값을 낮추거나 고정 범위를 좁히세요.",
                    "Outlier 이미지를 별도로 확인하고 필요하면 제외하거나 별도 preset을 사용하세요.",
                ],
            )
        )

    if v_mean is not None:
        if v_mean > 220:
            advice.append(
                _advice(
                    "WARNING",
                    "밝기 과다 가능성",
                    f"평균 Value가 {v_mean:.2f}입니다. 밝은 영역 clipping 가능성이 있습니다.",
                    [
                        "Intensity를 낮추세요.",
                        "Gamma를 1.05~1.20 방향으로 조정해 밝은 영역을 완화하세요.",
                        "Contrast를 1.0 이하로 낮춰보세요.",
                    ],
                )
            )
        elif v_mean < 60:
            advice.append(
                _advice(
                    "INFO",
                    "밝기 부족 가능성",
                    f"평균 Value가 {v_mean:.2f}입니다. 결과가 어둡게 보일 수 있습니다.",
                    [
                        "Intensity를 올리세요.",
                        "Gamma를 0.8~0.95 방향으로 조정해 중간톤을 밝게 만드세요.",
                    ],
                )
            )

    if rgb_std_r is not None and rgb_std_g is not None and rgb_std_b is not None:
        max_std = max(rgb_std_r, rgb_std_g, rgb_std_b)
        if max_std > 40:
            dominant = ["R", "G", "B"][[rgb_std_r, rgb_std_g, rgb_std_b].index(max_std)]
            advice.append(
                _advice(
                    "INFO",
                    f"{dominant} 채널 편차가 큼",
                    f"이미지 간 {dominant} 평균값 편차가 {max_std:.2f}입니다. 특정 색상 bias가 이미지별로 다를 수 있습니다.",
                    [
                        f"{dominant} Scale을 고정값으로 크게 조정하기보다 0.05 단위로 미세 조정하세요.",
                        "Dataset을 색상 계열별로 나누어 별도 preset을 적용하는 것도 고려하세요.",
                    ],
                )
            )

    if outliers:
        advice.append(
            _advice(
                "WARNING",
                "Saturation outlier 존재",
                f"채도 기준 outlier가 {len(outliers)}개 감지되었습니다.",
                [
                    "Outlier 이미지를 시각적으로 확인하세요.",
                    "전체 preset을 바꾸기 전에 outlier만 별도 처리할지 판단하세요.",
                    "Outlier가 실제 결함/특수 조건 이미지라면 별도 그룹으로 분리하는 것이 좋습니다.",
                ],
            )
        )

    if not advice:
        advice.append(
            _advice(
                "GOOD",
                "Dataset QA 상태 양호",
                "현재 통계 기준으로 큰 색감 불안정성은 보이지 않습니다.",
                [
                    "현재 파라미터를 baseline preset으로 저장하세요.",
                    "Evaluation 탭에서 GT 기준 Delta E를 추가 확인하세요.",
                ],
            )
        )

    return advice
