import streamlit as st
from src.advisor import generate_dataset_qa_advice, generate_evaluation_advice

st.set_page_config(page_title='Colorization QA Tool', layout='wide')
st.title('Colorization QA Tool')

st.warning('Temporary recovery build applied. Full UI restoration pending.')

st.markdown('## Evaluation Recommendation Preview')
example_eval = [{
    'name': 'sample.png',
    'hist_mean': 0.61,
    'delta_e_mean': 15.3,
    'delta_e_max': 29.1,
    'psnr': 18.2,
}]

for item in generate_evaluation_advice(example_eval):
    st.markdown(f"### [{item['level']}] {item['title']}")
    st.write(item['reason'])
    for action in item['actions']:
        st.write(f'- {action}')

st.markdown('---')

st.markdown('## Dataset QA Recommendation Preview')
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

for item in generate_dataset_qa_advice(summary, outliers):
    st.markdown(f"### [{item['level']}] {item['title']}")
    st.write(item['reason'])
    for action in item['actions']:
        st.write(f'- {action}')
