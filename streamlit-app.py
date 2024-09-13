import streamlit as st
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt

# 사용자 정의 HTML과 CSS 추가
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        color: #4B0082;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, rgba(34,193,195,1) 0%, rgba(253,187,45,1) 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 28px;
        color: #663399;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 20px;
    }
    .stRadio > div {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
    }
    .stSelectbox > div {
        background-color: #e6e6fa;
        border-radius: 10px;
    }
    .stSlider > div {
        background-color: #ffe4e1;
        border-radius: 10px;
    }
    .chart-container {
        margin-top: 20px;
        padding: 10px;
        background-color: #fff5ee;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 메인 제목
st.markdown('<div class="main-header">👋 Matlab to Python 데모 앱</div>', unsafe_allow_html=True)

# 사이드바 info
st.sidebar.info("""
Matlab -> Python 변환 프로젝트 데모 앱
""")


# 사이드바 옵션 설정
option = st.sidebar.radio(
    "🧚🏻‍♂️ 수행할 태스크를 선택해 주세요!",
    ('Matlab -> Python 코드 변환', '파이썬 함수의 수식화', 'GUI 예시')
)

# Matlab -> Python 코드 변환 섹션
if option == 'Matlab -> Python 코드 변환':
    st.markdown('<div class="sub-header">🐍 Matlab -> Python 코드 변환</div>', unsafe_allow_html=True)
    transform_option = st.selectbox(
        "🧚🏻‍♂️ 변환할 기능 선택",
        ('푸리에 변환', '역푸리에 변환', '신호처리-내삽', '신호처리-외삽')
    )
    
    # 코드 예제 사전 정의
    code_examples = {
        '푸리에 변환': {
            'Matlab 코드': '''
% 푸리에 변환 Matlab 코드
t = 0:0.002:1;
signal = sin(2*pi*5*t) + 0.5*sin(2*pi*10*t);
fft_result = fft(signal);
''',
            'Python 코드': '''
# Fourier Transform Python 코드
import numpy as np
from scipy.fft import fft

t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
fft_result = fft(signal)
'''
        },
        '역푸리에 변환': {
            'Matlab 코드': '''
% 역푸리에 변환 Matlab 코드
ifft_result = ifft(fft_result);
''',
            'Python 코드': '''
# Inverse Fourier Transform Python 코드
from scipy.fft import ifft

ifft_result = ifft(fft_result)
'''
        },
        '신호처리-내삽': {
            'Matlab 코드': '''
% 신호 처리 내삽 Matlab 코드
x = 0:10;
y = sin(x);
x_new = 0:0.1:10;
y_new = interp1(x, y, x_new, 'cubic');
''',
            'Python 코드': '''
# Signal Processing Interpolation Python 코드
import numpy as np
from scipy.interpolate import interp1d

x = np.linspace(0, 10, 11)
y = np.sin(x)
f_interpolate = interp1d(x, y, kind='cubic')
x_new = np.linspace(0, 10, 100)
y_new = f_interpolate(x_new)
'''
        },
        '신호처리-외삽': {
            'Matlab 코드': '''
% 신호 처리 외삽 Matlab 코드
spline_result = spline(x, y, 15);
''',
            'Python 코드': '''
# Signal Processing Extrapolation Python 코드
from scipy.interpolate import UnivariateSpline

spline = UnivariateSpline(x, y, s=0)
x_extrap = np.linspace(10, 15, 50)
y_extrap = spline(x_extrap)
'''
        }
    }
    
    # 선택된 기능의 Matlab과 Python 코드 표시
    if transform_option in code_examples:
        matlab_code = code_examples[transform_option]['Matlab 코드']
        python_code = code_examples[transform_option]['Python 코드']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matlab 코드")
            st.code(matlab_code, language='matlab')
        with col2:
            st.subheader("Python 코드")
            st.code(python_code, language='python')

# 파이썬 함수의 수식화 섹션
elif option == '파이썬 함수의 수식화':
    st.markdown('<div class="sub-header">Ω 파이썬 함수의 수식화</div>', unsafe_allow_html=True)
    
    formula_option = st.selectbox(
        "🧚🏻‍♂️ 수식화할 기능 선택",
        ('푸리에 변환', '역푸리에 변환')
    )
    
    formula_examples = {
        '푸리에 변환': {
            'Python 코드': '''
import numpy as np
from scipy.fft import fft

def perform_fourier_transform(signal):
    transformed_signal = fft(signal)
    return transformed_signal
''',
            'LaTeX 수식': r'''
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2 \pi}{N} k n} \quad \text{for} \quad k = 0, 1, 2, \dots, N-1
'''
        },
        '역푸리에 변환': {
            'Python 코드': '''
from scipy.fft import ifft

def perform_inverse_fourier_transform(transformed_signal):
    ifft_result = ifft(transformed_signal)
    return ifft_result
''',
            'LaTeX 수식': r'''
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j \frac{2 \pi}{N} k n} \quad \text{for} \quad n = 0, 1, 2, \dots, N-1
'''
        }
    }
    
    if formula_option in formula_examples:
        python_code = formula_examples[formula_option]['Python 코드']
        latex_formula = formula_examples[formula_option]['LaTeX 수식']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("파이썬 함수 코드")
            st.code(python_code, language='python')
        with col2:
            st.subheader("LaTeX 수식")
            st.latex(latex_formula)

# GUI 예시 섹션
elif option == 'GUI 예시':
    st.markdown('<div class="sub-header">📊 GUI 예시</div>', unsafe_allow_html=True)
    gui_option = st.selectbox(
        "🧚🏻‍♂️ 수행할 기능 선택",
        ('푸리에 변환', '역푸리에 변환', '신호처리-내삽', '신호처리-외삽')
    )
    st.markdown("👇 파라미터 값 선택")
    if gui_option in ['푸리에 변환', '역푸리에 변환']:
        freq1 = st.number_input("첫 번째 사인파 주파수 (Hz)", min_value=1, max_value=50, value=5)
        freq2 = st.number_input("두 번째 사인파 주파수 (Hz)", min_value=1, max_value=50, value=10)
        amplitude1 = st.number_input("첫 번째 사인파 진폭", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        amplitude2 = st.number_input("두 번째 사인파 진폭", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        sample_rate = st.number_input("샘플 수", min_value=100, max_value=2000, value=500, step=100)
        
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        signal = amplitude1 * np.sin(2 * np.pi * freq1 * t) + amplitude2 * np.sin(2 * np.pi * freq2 * t)
        
        if gui_option == '푸리에 변환':
            transformed_signal = fft(signal)
            title = 'Fourier Transform Result'
            plot_data = np.abs(transformed_signal)
        elif gui_option == '역푸리에 변환':
            transformed_signal = ifft(fft(signal))
            title = 'Inverse Fourier Transform Result'
            plot_data = np.real(transformed_signal)
        
        # 차트 시각화
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(t, signal, label='Original Signal')
        ax.plot(t, plot_data, label='Transformed Signal')
        ax.set_title(title)
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif gui_option == '신호처리-내삽':
        num_points = st.slider("원본 데이터 포인트 수", min_value=5, max_value=20, value=10)
        x = np.linspace(0, 10, num_points)
        y = np.sin(x)
        x_new = np.linspace(0, 10, 100)
        f_interpolate = interp1d(x, y, kind='cubic')
        y_new = f_interpolate(x_new)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Original Data')
        ax.plot(x_new, y_new, '-', label='Interpolated Data')
        ax.set_title('Signal Processing - Interpolation Result')
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif gui_option == '신호처리-외삽':
        num_points = st.slider("원본 데이터 포인트 수", min_value=5, max_value=20, value=10)
        x = np.linspace(0, 10, num_points)
        y = np.sin(x)
        spline = UnivariateSpline(x, y, s=0)
        x_extrap = np.linspace(0, 15, 150)
        y_extrap = spline(x_extrap)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Original Data')
        ax.plot(x_extrap, y_extrap, '--', label='Extrapolated Data')
        ax.set_title('Signal Processing - Extrapolation Result')
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# 추가 정보 또는 설명 (선택 사항)
st.sidebar.markdown("""
---
**🔖  참고 사항**

- 이 웹앱은 단순한 예시입니다. 😄
- 고객과 UI 세부 협의 후 PC용 GUI 구현해 드리겠습니다! 👩🏻‍💻🙇🏻‍♂️
""")