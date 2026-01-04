import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="K-콘텐츠 정책 효과 시뮬레이터",
    page_icon="📈",
    layout="wide"
)

# --- 한글 폰트 설정 (Streamlit Cloud 호환) ---
# 리눅스 환경(Cloud)에서 한글 폰트가 깨지는 것을 방지하기 위한 설정입니다.
def setup_font():
    if os.name == 'posix':  # 리눅스 환경 (Streamlit Cloud)
        plt.rc('font', family='NanumGothic')
    else:  # 윈도우 환경 (로컬 실행 시)
        plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# --- 메인 헤더 ---
st.title("📈 K-콘텐츠 정책 효과 인과분석 수리 모델링")
st.markdown("""
이 대시보드는 **'K-콘텐츠 팬덤 경제의 이중 전선'** 연구를 기반으로, 
경제 정책 변수(투입 예산, 마찰계수, 누수율)가 산업 성장에 미치는 인과 관계를 시뮬레이션합니다.
""")

# --- 사이드바: 변수 컨트롤러 ---
with st.sidebar:
    st.header("🎛️ 정책 변수 설정")
    st.info("아래 슬라이더를 조절하여 시나리오를 변경해보세요.")
    
    # 변수 1: 정책 마찰계수
    mu = st.slider(
        "1. 정책 마찰계수 ($\mu$)", 
        min_value=0.0, max_value=0.9, value=0.6, step=0.1,
        help="0에 가까울수록 자금 집행이 원활하며, 1에 가까울수록 규제나 행정 비효율로 자금이 묶입니다."
    )
    
    # 변수 2: 플랫폼 수수료 (누수율)
    lambda_plat = st.slider(
        "2. 플랫폼 수수료율 ($\lambda_{plat}$)", 
        min_value=0, max_value=50, value=30, step=5,
        help="구글/애플 등 플랫폼에 지불하는 수수료 비율입니다. (단위: %)"
    )
    
    # 변수 3: 투자 전환 강도
    innovation = st.slider(
        "3. 구조 개혁 강도 (Investment Intensity)", 
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help="플랫폼/IP 확보를 위한 투자 강도입니다. 높을수록 초기 비용은 크지만 장기 성장률이 높습니다."
    )

# --- 탭 구성 ---
tab1, tab2, tab3 = st.tabs(["💰 1.4조 펀드 효율성", "📉 가치 사슬 누수", "🚀 J-커브 성장 예측"])

# --- TAB 1: 1.4조 원 펀드와 마찰계수 모델 ---
with tab1:
    st.subheader("1. 정책 마찰계수($\mu$)에 따른 유효 투자액 분석")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        total_fund = 1.4  # 1.4조 원
        effective_fund = total_fund * (1 - mu)
        loss = total_fund - effective_fund
        
        st.metric(label="총 조성 펀드 규모", value="1.4 조 원")
        st.metric(label="실제 시장 유효 투자액 ($I_{eff}$)", value=f"{effective_fund:.2f} 조 원", delta=f"-{loss:.2f} 조 원 (마찰 손실)", delta_color="inverse")
        
        st.markdown(f"""
        **분석:**
        현재 마찰계수 **{mu}** 설정 시, 
        전체 펀드의 **{(1-mu)*100:.0f}%** 만이 실제 기업에 도달합니다.
        나머지 자금은 투자처를 찾지 못해 미집행 상태(돈맥경화)로 남습니다.[1, 3]
        """)

    with col2:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        x_mu = np.linspace(0, 1, 100)
        y_eff = total_fund * (1 - x_mu)
        
        ax1.plot(x_mu, y_eff, color='#FF5733', linewidth=3, label='유효 투자액')
        ax1.scatter(mu, effective_fund, color='blue', s=150, zorder=5)
        ax1.annotate(f'현재 설정 ($\mu={mu}$)', (mu, effective_fund), 
                     xytext=(mu+0.1, effective_fund+0.3), 
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax1.fill_between(x_mu, y_eff, alpha=0.1, color='#FF5733')
        ax1.set_xlabel("정책 마찰계수 ($\mu$)")
        ax1.set_ylabel("유효 투자액 (조 원)")
        ax1.set_title("마찰계수 증가에 따른 투자 효율성 감소")
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

# --- TAB 2: 가치 사슬 누수 구조 (Waterfall) ---
with tab2:
    st.subheader("2. 산업별 가치 사슬 누수($\lambda$) 구조 시각화")
    
    revenue = 100
    fee = -(revenue * lambda_plat / 100)
    marketing = -30  # 마케팅비 가정
    production = -35 # 제작비 가정
    operating_profit = revenue + fee + marketing + production
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    categories = ['매출액', '플랫폼 수수료', '마케팅 비용', '제작/운영비', '최종 영업이익']
    values = [revenue, fee, marketing, production, operating_profit]
    
    # Waterfall 차트 로직
    bottoms = [0, revenue, revenue+fee, revenue+fee+marketing, 0]
    colors =
    
    bars = ax2.bar(categories, values, bottom=bottoms, color=colors, edgecolor='black', alpha=0.8)
    
    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = bar.get_y() + height / 2
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.0f}', 
                ha='center', va='center', color='white', fontweight='bold')

    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title(f"수수료 {lambda_plat}% 적용 시 수익 구조 분석")
    st.pyplot(fig2)
    
    st.markdown(f"""
    **해석:**
    매출 100이 발생해도 플랫폼 수수료($\lambda_{{plat}}$)로 **{abs(fee):.0f}** 가 즉시 유출됩니다.
    여기에 마케팅비와 제작비를 제외하면 최종 영업이익은 **{operating_profit:.1f}** 수준입니다.[4, 5]
    """)

# --- TAB 3: 동태적 분석 (J-Curve) ---
with tab3:
    st.subheader("3. 정책 시차와 J-커브(J-Curve) 효과 시뮬레이션")
    
    t = np.linspace(0, 10, 100)
    
    # 시나리오 1: 현상 유지 (단기 성과는 좋으나 장기 침체)
    y_status = 50 + 5 * np.log(t + 1)
    
    # 시나리오 2: 구조 개혁 (초기 비용 발생 후 급성장)
    y_reform = 50 - (innovation * 3) * t * np.exp(-0.8 * t) + (innovation * 0.8) * t**1.8 * (1 - np.exp(-0.3*t))
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t, y_status, 'gray', linestyle='--', linewidth=2, label='현행 유지 (콘텐츠 중심)')
    ax3.plot(t, y_reform, '#1F618D', linewidth=3, label=f'구조 개혁 (투자 강도 {innovation})')
    
    # 교차점(손익분기점) 표시
    idx = np.argwhere(np.diff(np.sign(y_reform - y_status))).flatten()
    if len(idx) > 0:
        cross_t = t[idx[-1]]
        cross_y = y_reform[idx[-1]]
        ax3.scatter(cross_t, cross_y, color='red', zorder=5)
        ax3.annotate(f'전환점 ({cross_t:.1f}년 후)', (cross_t, cross_y), 
                     xytext=(cross_t, cross_y+10), arrowprops=dict(facecolor='red', arrowstyle='->'))

    ax3.set_xlabel("시간 (년)")
    ax3.set_ylabel("국내 부가가치 창출액")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    st.pyplot(fig3)
    st.markdown("""
    **J-커브 이론 적용:**
    구조 개혁(플랫폼 육성) 초기에는 투자 비용으로 인해 성과가 하락하는 '죽음의 계곡' 구간이 발생합니다.
    하지만 일정 시간($t^*$)이 지나면 네트워크 효과에 의해 기하급수적인 성장을 달성함을 보여줍니다.[6]
    """)

st.divider()
st.caption("Developed for K-Content Economic Research | Powered by Streamlit & Python")