import streamlit as st
from src.advisor import generate_dataset_qa_advice, generate_evaluation_advice

st.set_page_config(page_title='Colorization QA Tool', layout='wide')
st.title('Colorization QA Tool')

uploaded = st.file_uploader(
    '이미지 업로드',
    type=['png', 'jpg', 'jpeg', 'bmp'],
    accept_multiple_files=True,
)

if uploaded:
    st.success(f'{len(uploaded)}개 이미지 업로드 완료')


tab_gray, tab_color, tab_eval, tab_qa = st.tabs([
    'Grayscale Export',
    'Colorization',
    'Evaluation',
    'Dataset QA'
])

with tab_gray:
    st.subheader('Grayscale Export')
    st.write('실사 RGB 이미지를 grayscale benchmark dataset으로 변환합니다.')

    if uploaded:
        cols = st.columns(min(4, len(uploaded)))
        for idx, file in enumerate(uploaded[:4]):
            with cols[idx % len(cols)]:
                st.image(file, caption=file.name, use_container_width=True)

with tab_color:
    st.subheader('Colorization')

    c1, c2 = st.columns(2)

    with c1:
        st.slider('R Scale', 0.0, 3.0, 1.0, 0.05)
        st.slider('G Scale', 0.0, 3.0, 1.0, 0.05)
        st.slider('B Scale', 0.0, 3.0, 1.0, 0.05)

    with c2:
        st.slider('Gamma', 0.1, 3.0, 1.0, 0.05)
        st.slider('Contrast', 0.1, 3.0, 1.0, 0.05)
        st.slider('Saturation', 0.0, 3.0, 1.0, 0.05)

    st.info('기존 Colorization UI 복구 단계 진행 중')

with tab_eval:
    st.subheader('Evaluation')

    example_eval = [{
        'name': 'sample.png',
        'hist_mean': 0.61,
        'delta_e_mean': 15.3,
        'delta_e_max': 29.1,
        'psnr': 18.2,
    }]

    st.markdown('### Evaluation Metrics')
    st.table(example_eval)

    st.markdown('### QA Recommendation')

    for item in generate_evaluation_advice(example_eval):
        level = item['level']

        if level == 'CRITICAL':
            st.error(f"[{level}] {item['title']}")
        elif level == 'WARNING':
            st.warning(f"[{level}] {item['title']}")
        elif level == 'GOOD':
            st.success(f"[{level}] {item['title']}")
        else:
            st.info(f"[{level}] {item['title']}")

        st.write(item['reason'])

        for action in item['actions']:
            st.write(f'- {action}')

with tab_qa:
    st.subheader('Dataset QA')

    summary = {
        's_mean': {'mean': 140.0, 'std': 40.0, 'min': 10.0, 'max': 255.0},
        'v_mean': {'mean': 225.0, 'std': 15.0, 'min': 50.0, 'max': 255.0},
        'rgb_mean_r': {'mean': 170.0, 'std': 55.0, 'min': 0.0, 'max': 255.0},
        'rgb_mean_g': {'mean': 140.0, 'std': 20.0, 'min': 0.0, 'max': 255.0},
        'rgb_mean_b': {'mean': 120.0, 'std': 15.0, 'min': 0.0, 'max': 255.0},
    }

    outliers = [
        {'name': 'sample_outlier.png', 'z_score': 3.2}
    ]

    st.markdown('### Dataset Statistics')
    st.json(summary)

    st.markdown('### Dataset QA Insight')

    for item in generate_dataset_qa_advice(summary, outliers):
        level = item['level']

        if level == 'CRITICAL':
            st.error(f"[{level}] {item['title']}")
        elif level == 'WARNING':
            st.warning(f"[{level}] {item['title']}")
        elif level == 'GOOD':
            st.success(f"[{level}] {item['title']}")
        else:
            st.info(f"[{level}] {item['title']}")

        st.write(item['reason'])

        for action in item['actions']:
            st.write(f'- {action}')
