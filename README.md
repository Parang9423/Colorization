# Colorization

Streamlit 기반 이미지 컬러라이징 보정 툴.

## Features

- HSV 기반 색상 보정
- ROI 기반 영역 보정
- Gamma 조절
- PNG 저장
- JSON 파라미터 저장
- Original / Adjusted 비교 UI

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Stack

- Streamlit
- OpenCV
- NumPy
- Pillow

## Future Work

- Brush 기반 ROI 선택
- SAM / YOLO Segmentation 연동
- Color palette recommendation
- LAB colorspace support
- Dataset management
