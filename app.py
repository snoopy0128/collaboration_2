import streamlit as st
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------- DB 연결 -------------------
@st.cache_resource(show_spinner="DB 연결 중...")
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='12341234',
        database='car_dashboard',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

conn = get_connection()

st.title("🚗 서울 자동차 등록 현황 대시보드")

menu = st.sidebar.radio("메뉴 선택",
    [
        "홈 (규모별 현황)",
        "친환경 자동차 등록 현황",
        "전기차 vs 전체 승용차 분석 및 전기차 비중 예측",
        "전기차 분류 모델",
        "CCTV vs 사고 예측 모델"
    ]
)

# ------------------- 데이터 조회 함수 -------------------
def fetch_query(query):
    with conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
    return pd.DataFrame(results)

# ------------------- 홈 (규모별 현황) -------------------
if menu == "홈 (규모별 현황)":
    st.markdown("### 2022~2025년 서울 승용차 규모별 등록 현황")

    try:
        # 규모별 데이터 로드
        df_size = fetch_query("SELECT 연도, 규모, 승용 FROM seoul_size_registration WHERE 시도='서울' ORDER BY 연도, 규모")

        if df_size.empty:
            st.warning("규모별 데이터를 불러오지 못했습니다.")
            st.stop()

        # 연도별 총합 계산
        total_by_year = df_size.groupby('연도')['승용'].sum().reset_index()
        latest_year = total_by_year['연도'].max()
        latest_total = total_by_year[total_by_year['연도'] == latest_year]['승용'].values[0]

        # 변화량 계산
        total_by_year['변화량'] = total_by_year['승용'].diff()

        # 최신 연도 요약
        st.subheader(f"📅 {int(latest_year)}년 서울 전체 승용차 등록")
        col1, col2 = st.columns(2)
        with col1:
            delta = total_by_year[total_by_year['연도'] == latest_year]['변화량'].values[0] if len(total_by_year) > 1 else 0
            st.metric("총 등록 대수", f"{int(latest_total):,}", f"{int(delta):+,}대")
        with col2:
            st.metric("데이터 기준 연도", int(latest_year))

        # 1. 연도별 전체 등록 대수 추이 막대 그래프 (y축 고정으로 변동 강조)
        st.subheader("📊 연도별 전체 승용차 등록 대수 변화 (미세 변동 확대)")
        fig_bar, ax_bar = plt.subplots(figsize=(11, 6))

        # 증가/감소 색상 구분
        colors = ['#4CAF50' if x >= 0 else '#F44336' for x in total_by_year['변화량'].fillna(0)]

        bars = ax_bar.bar(total_by_year['연도'], total_by_year['승용'], color=colors, edgecolor='black', width=0.6)

        # 막대 위에 숫자 + 변화량 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            change = total_by_year['변화량'].iloc[i] if i > 0 else 0
            change_str = f"{int(change):+,}대" if i > 0 else "기준"
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 500,
                        f'{int(height):,}\n{change_str}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11, color='black')

        ax_bar.set_title('서울 전체 승용차 등록 대수 변화 (2022~2025)', fontsize=16, pad=20)
        ax_bar.set_xlabel('연도', fontsize=12)
        ax_bar.set_ylabel('등록 대수', fontsize=12)
        ax_bar.grid(alpha=0.3, axis='y', linestyle='--')

        # y축 범위 고정
        ax_bar.set_ylim(2760000, 2780000)

        # y축 눈금 간격 조정 (2만 단위)
        ax_bar.set_yticks(np.arange(2760000, 2780001, 20000))
        ax_bar.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        st.pyplot(fig_bar)

        # 2. 규모별 파이차트
        st.subheader("서울 승용차 규모별 구성 비율 (2025)")
        sizes = df_size.groupby('규모')['승용'].sum()

        fig_pie, ax_pie = plt.subplots(figsize=(9, 9))
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=sizes.index, autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Pastel1(range(len(sizes))), textprops={'fontsize': 13})
        ax_pie.set_title('규모별 비중 (2025)', fontsize=18, pad=20)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        st.pyplot(fig_pie)

        # 3. 상세 테이블
        st.markdown("---")
        st.subheader("📋 2022~2025년 규모별 등록 대수 상세")
        pivot_table = df_size.pivot(index='연도', columns='규모', values='승용').fillna(0).astype(int)
        pivot_table['합계'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_index(ascending=False)

        styled_table = pivot_table.style\
            .format('{:,}')\
            .set_properties(**{'text-align': 'center', 'font-size': '14px'})\
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]},
            ])\
            .bar(subset=['합계'], color='#a8e6cf')

        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        # 결론
        st.info("""
        📊 **요약**  
        • 서울 전체 승용차는 2022~2025년간 **약간의 정체 → 소폭 감소** 추세  
        • 중형 + 대형이 여전히 **85% 이상** 압도적 비중  
        • 소형·경형은 지속적으로 줄어드는 중  

        👈 왼쪽 메뉴에서 친환경차 증가 추이와 미래 예측을 확인하세요!
        """)

        st.caption("데이터 출처: 국토교통부 승용차 등록 통계 (2025년 포함 최신)")

    except Exception as e:
        st.error(f"홈 화면 로드 중 오류: {e}")

# ------------------- 친환경 자동차 등록 현황 -------------------
elif menu == "친환경 자동차 등록 현황":
    st.header("🌿 서울 친환경 자동차 등록 현황")
    st.markdown("**2022~2024년 전기차 · 하이브리드 · 수소차 보급 추이**")

    try:
        df = fetch_query("SELECT * FROM seoul_fuel_registration WHERE 시도='서울' ORDER BY 연도")

        if df.empty:
            st.warning("데이터를 불러오지 못했습니다.")
            st.stop()

        # 테이블
        st.subheader("연도별 등록 대수")
        display_df = df[['연도', '전기_승용', '하이브리드_승용', '수소_승용']].rename(columns={
            '전기_승용': '전기차', '하이브리드_승용': '하이브리드', '수소_승용': '수소차'
        })
        st.dataframe(display_df.style.format('{:,}'), use_container_width=True, hide_index=True)  # 인덱스 숨김

        # 핵심 메트릭
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        st.subheader(f"{int(latest['연도'])}년 증가량")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전기차", f"{int(latest['전기_승용']):,}", f"+{int(latest['전기_승용'] - prev['전기_승용']):,}대" if prev is not None else "")
        with col2:
            st.metric("하이브리드", f"{int(latest['하이브리드_승용']):,}", f"+{int(latest['하이브리드_승용'] - prev['하이브리드_승용']):,}대" if prev is not None else "")
        with col3:
            st.metric("수소차", f"{int(latest['수소_승용']):,}", f"+{int(latest['수소_승용'] - prev['수소_승용']):,}대" if prev is not None else "")

        # 추이 그래프
        st.subheader("2022~2024년 증가 추이")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(df['연도'], df['전기_승용'], marker='s', linewidth=5, markersize=14, label='전기차', color='#1f77b4')
        ax.plot(df['연도'], df['하이브리드_승용'], marker='^', linewidth=5, markersize=14, label='하이브리드', color='#ff7f0e')
        ax.plot(df['연도'], df['수소_승용'], marker='D', linewidth=5, markersize=14, label='수소차', color='#2ca02c')
        ax.set_title('서울 친환경 자동차 등록 추이', fontsize=16)
        ax.set_xlabel('연도')
        ax.set_ylabel('등록 대수')
        ax.legend(fontsize=13)
        ax.grid(alpha=0.3)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        for i, row in df.iterrows():
            ax.text(row['연도'], row['전기_승용'], f"{int(row['전기_승용']):,}", ha='center', va='bottom', fontweight='bold', color='#1f77b4')
            ax.text(row['연도'], row['하이브리드_승용'], f"{int(row['하이브리드_승용']):,}", ha='center', va='bottom', fontweight='bold', color='#ff7f0e')
            ax.text(row['연도'], row['수소_승용'], f"{int(row['수소_승용']):,}", ha='center', va='bottom', fontweight='bold', color='#2ca02c')
        st.pyplot(fig)

        st.success("""
        🌱 **요약**  
        • 하이브리드가 가장 많지만 증가세 둔화  
        • 전기차가 가장 빠르게 성장 중  
        • 수소차는 초기 단계이나 꾸준히 증가
        """)

    except Exception as e:
        st.error(f"오류: {e}")

# ------------------- 전기차 & 충전기 분석 -------------------
elif menu == "전기차 & 충전기 분석":
    st.header("🔋 서울 전기차 등록 vs 충전기 인프라 분석")
    st.markdown("**2022~2024년 누적 데이터 기반** (충전기: 환경부, 전기차: 국토부 승용 기준)")

    try:
        # 2024년까지 강제 제한 (2025년 데이터 완전 제외)
        query = """
        SELECT 
            f.연도,
            f.전기_승용 AS 누적_전기차,
            COALESCE(c.누적_충전기, 0) AS 누적_충전기
        FROM seoul_fuel_registration f
        LEFT JOIN seoul_chargers c ON f.연도 = c.year
        WHERE f.시도 = '서울'
          AND f.연도 BETWEEN 2022 AND 2024
        ORDER BY f.연도
        """
        df = fetch_query(query)

        if df.empty or len(df) < 3:
            st.warning("데이터 부족 (2022~2024년 데이터 필요). JOIN 또는 테이블 확인.")
            st.stop()

        if df['누적_충전기'].sum() == 0:
            st.warning("충전기 데이터가 없음. seoul_chargers 테이블 데이터 확인.")
            st.stop()

        # 선형 회귀 모델
        model = LinearRegression()
        X = df[['누적_충전기']]
        y = df['누적_전기차']
        model.fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)

        # 충전기 1기당 전기차 비율
        df['충전기1기당_전기차'] = df['누적_전기차'] / df['누적_충전기'].replace(0, 1)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("충전기 1기 증가 시", f"+{slope:.3f}대", "전기차 등록 증가 (평균)")
            st.metric("현재 평균 비율 (2024년)", f"{df['충전기1기당_전기차'].iloc[-1]:.2f}대", "충전기 1기당 전기차")

        with col2:
            st.metric("모델 설명력 (R²)", f"{r2:.6f}")
            last_year = df['연도'].iloc[-1]
            st.metric(f"{last_year}년 누적 충전기", f"{int(df['누적_충전기'].iloc[-1]):,}기")

        # 그래프 1: 누적 추이
        st.subheader("누적 추이 (2022~2024)")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df['연도'], df['누적_충전기'], marker='o', label='누적 충전기', linewidth=3, color='blue')
        ax1.plot(df['연도'], df['누적_전기차'], marker='s', label='누적 전기차', linewidth=3, color='green')
        ax1.set_title('서울 누적 충전기 vs 전기차 등록 추이 (2022~2024)')
        ax1.set_ylabel('대수')
        ax1.set_xlabel('연도')
        ax1.legend()
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

        # 그래프 2: 비율 추이
        st.subheader("충전기 1기당 전기차 대수 추이")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df['연도'], df['충전기1기당_전기차'], marker='D', color='purple', linewidth=3, markersize=10)
        ax2.set_title('충전기 1기당 지원 가능한 전기차 대수 변화 (2022~2024)')
        ax2.set_ylabel('전기차 대수 / 충전기 1기')
        ax2.set_xlabel('연도')
        ax2.grid(alpha=0.3)
        for i, row in df.iterrows():
            ax2.text(row['연도'], row['충전기1기당_전기차'] + 0.01, f"{row['충전기1기당_전기차']:.2f}", 
                     ha='center', fontweight='bold')
        st.pyplot(fig2)

        # 회귀 산점도
        st.subheader("상관 분석 및 회귀 모델")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(df['누적_충전기'], df['누적_전기차'], s=150, color='darkblue', zorder=5)
        x_line = np.array([df['누적_충전기'].min(), df['누적_충전기'].max()])
        y_line = model.predict(x_line.reshape(-1, 1))
        ax3.plot(x_line, y_line, color='red', linewidth=3, label=f'회귀선 (기울기={slope:.3f})')
        for i, row in df.iterrows():
            ax3.text(row['누적_충전기'] + 600, row['누적_전기차'], str(row['연도']), fontsize=12, fontweight='bold')
        ax3.set_xlabel('누적 충전기 대수')
        ax3.set_ylabel('누적 전기차 등록 대수')
        ax3.set_title(f'누적 상관 분석 (R² = {r2:.6f}, 2022~2024)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        st.info("JOIN 실패 또는 데이터 부족. seoul_chargers 테이블의 'year'와 '누적_충전기' 확인.")
        
# ------------------- 전기차 vs 전체 승용차 분석 및 비중 예측 -------------------
elif menu == "전기차 vs 전체 승용차 분석 및 전기차 비중 예측":
    st.header("🔍 전기차와 전체 승용차 관계 분석")

    tab1, tab2 = st.tabs(["📊 현재 추이 분석", "🚀 미래 전기차 비중 예측 (2026~2030)"])

    with tab1:
        st.markdown("**2022~2025년 데이터로 본 '전기차가 서울 자동차 시장에 미치는 영향'**")

        try:
            total_df = fetch_query("""
                SELECT 연도, SUM(승용) AS 총_승용차
                FROM seoul_size_registration
                GROUP BY 연도
                ORDER BY 연도
            """)
            ev_df = fetch_query("SELECT 연도, 전기_승용 AS 전기차 FROM seoul_fuel_registration ORDER BY 연도")
            df = pd.merge(total_df, ev_df, on='연도')
            df = df.sort_values('연도').reset_index(drop=True)
            df['총_승용차'] = df['총_승용차'].astype(int)
            df['전기차'] = df['전기차'].astype(int)

            st.subheader("📅 연도별 등록 대수 요약")
            display_df = df.copy()
            display_df['전기차 비율 (%)'] = (display_df['전기차'] / display_df['총_승용차'] * 100).round(2)
            st.dataframe(
                display_df.rename(columns={
                    '연도': '연도', '총_승용차': '전체 자동차', '전기차': '전기차', '전기차 비율 (%)': '전기차 비율 (%)'
                }).style.format({'전체 자동차': '{:,}', '전기차': '{:,}', '전기차 비율 (%)': '{:.2f}%'}),
                use_container_width=True, hide_index=True
            )

            st.subheader("🔑 한눈에 보는 핵심 포인트")
            col1, col2, col3 = st.columns(3)
            latest_year = df['연도'].iloc[-1]
            latest_ev_ratio = (df['전기차'].iloc[-1] / df['총_승용차'].iloc[-1] * 100)
            with col1:
                st.metric("📈 전기차 비율 (2025년)", f"{latest_ev_ratio:.2f}%",
                          delta=f"{latest_ev_ratio - (df['전기차'].iloc[-2] / df['총_승용차'].iloc[-2] * 100):.2f}%p 증가")
            with col2:
                st.metric("⬆️ 전기차 증가량", f"{df['전기차'].iloc[-1] - df['전기차'].iloc[-2]:,}대", delta="2024→2025년")
            with col3:
                st.metric("📊 전체 자동차 변화", f"{df['총_승용차'].iloc[-1] - df['총_승용차'].iloc[-2]:+,}대", delta="2024→2025년")

            st.subheader("📊 전체 자동차 vs 전기차 추이 비교")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.set_xlabel('연도', fontsize=12)
            ax1.set_ylabel('전체 자동차 대수', color='gray', fontsize=12)
            ax1.plot(df['연도'], df['총_승용차'], marker='o', linewidth=4, markersize=10, color='gray', label='전체 자동차')
            ax1.tick_params(axis='y', labelcolor='gray')
            ax1.grid(alpha=0.3)
            ax2 = ax1.twinx()
            ax2.set_ylabel('전기차 대수', color='green', fontsize=12)
            ax2.plot(df['연도'], df['전기차'], marker='s', linewidth=5, markersize=12, color='green', label='전기차')
            ax2.tick_params(axis='y', labelcolor='green')
            ax1.set_title('서울 전체 자동차는 거의 그대로, 전기차는 꾸준히 증가!', fontsize=16, pad=20)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            st.pyplot(fig)

            st.subheader("💡 쉽게 풀어쓴 해석")
            st.success("""
            🎯 **핵심 결론**:  
            서울은 전체 자동차 수가 거의 변하지 않거나 조금 줄고 있는데,  
            **전기차만 꾸준히 늘고 있어요!**

            ✅ **무슨 의미?**  
            • 사람들이 새 차를 살 때 **전기차를 더 많이 선택**하고 있다는 뜻  
            • 전체 시장이 줄어도 전기차가 그 빈자리를 채우고 있음  
            • 앞으로 전기차 비율이 점점 더 높아질 가능성이 큽니다!

            🌱 전기차가 서울의 자동차 시장을 새롭게 바꾸고 있어요!
            """)
            st.caption("데이터 출처: 국토교통부 승용차 등록 통계")

        except Exception as e:
            st.error(f"현재 추이 분석 중 오류: {e}")
            
# ------------------- 전기차 비중 예측 -------------------
    with tab2:
        st.markdown("**2023~2025년 월별 데이터 기반 선형회귀 예측**")

        try:
            query = """
            SELECT ym AS 연월, total_cars AS 전체, ev_cars AS 전기차, ev_ratio AS 비중
            FROM seoul_ev_ratio_monthly
            ORDER BY ym ASC
            """
            df = fetch_query(query)
            if df.empty:
                st.error("DB에서 데이터를 불러오지 못했습니다. 테이블(seoul_ev_ratio_monthly)을 확인하세요.")
                st.stop()

            df['연월'] = df['연월'].astype(int)
            X = df[['전기차']].values
            y = df['비중'].values * 100

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            r2_test = r2_score(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)

            st.subheader("모델 성능 (훈련/테스트 분리)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² (테스트)", f"{r2_test:.4f}")
            with col2:
                st.metric("MAE (테스트)", f"{mae_test:.2f}")
            with col3:
                st.metric("훈련 데이터 크기", f"{len(X_train)} / {len(X)}")

            st.subheader("시나리오 설정")
            col1, col2 = st.columns(2)
            with col1:
                annual_ev_increase = st.slider("연간 전기차 등록 증가량 (대)", 10000, 60000, 25000, 1000)
            with col2:
                linkage_ratio = st.slider("전기차 1대 증가 시 전체 승용차 증가 비율 (0~1)", 0.0, 1.0, 0.4, 0.05, format="%.2f")

            latest_row = df.loc[df['연월'].idxmax()]
            latest_ev = latest_row['전기차']
            latest_total = latest_row['전체']
            latest_ratio = latest_row['비중'] * 100

            future_years = np.arange(2026, 2031)
            future_ev = [latest_ev + annual_ev_increase * (yr - 2025) for yr in future_years]
            future_total = []
            current_total = latest_total
            for ev in future_ev:
                ev_increase = ev - latest_ev
                total_increase = ev_increase * linkage_ratio
                current_total += total_increase
                future_total.append(round(current_total))
            future_ratio = model.predict(np.array(future_ev).reshape(-1, 1))

            st.subheader("미래 예측 결과 (연도별)")
            pred_df = pd.DataFrame({
                '연도': future_years,
                '예상 전기차 등록 (대)': [f"{int(ev):,}" for ev in future_ev],
                '예상 전체 승용차 (대)': [f"{int(tot):,}" for tot in future_total],
                '예상 전기차 비중 (%)': [f"{r:.2f}" for r in future_ratio]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.subheader("그래프 (실제 + 예측)")
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.scatter(df['전기차'], y, color='blue', s=60, alpha=0.7, label='실제 데이터 (2023~2025)')
            x_min = df['전기차'].min()
            x_max = max(future_ev) + 20000
            x_range = np.linspace(x_min, x_max, 200)
            y_range = model.predict(x_range.reshape(-1, 1))
            ax.plot(x_range, y_range, color='red', linewidth=2.5, label='선형회귀 모델')
            ax.scatter(future_ev, future_ratio, color='green', s=150, marker='*', label='미래 예측 (2026~2030)')
            ax.set_title('서울 전기차 등록대수 vs 비중 (전기차 증가 → 전체 자동 연동)', fontsize=14)
            ax.set_xlabel('전기차 등록대수 (대)', fontsize=12)
            ax.set_ylabel('전기차 비중 (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.2f}'))
            st.pyplot(fig)

            st.info(f"""
            📊 **2025년 11월 기준**  
            • 전기차 등록: {latest_ev:,}대  
            • 전체 승용차: {latest_total:,}대  
            • 전기차 비중: {latest_ratio:.2f}%  
            """)

        except Exception as e:
            st.error(f"미래 비중 예측 오류: {str(e)}")
        
# ------------------- 전기차 분류 모델 -------------------
elif menu == "전기차 분류 모델":
    st.header(" 배기량·연비 기반 차종 분류 모델")
    st.markdown("""
    **배기량(cc)과 연비(km/L)를 입력하면 차종(일반/전기차/하이브리드)을 예측**하는 분류 모델입니다.  
    RandomForestClassifier를 GridSearchCV로 최적화한 모델을 사용합니다.
    """)

    try:
        # 1. 데이터 로드 (CSV 파일이 프로젝트 폴더에 있다고 가정)
        df_raw = pd.read_csv('전기차분류.csv')
        
        # 불필요한 열(no 등) 제거
        if 'no' in df_raw.columns:
            df_raw = df_raw.drop(columns=['no'])
        
        st.subheader("📊 원본 데이터 미리보기")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # 2. Label Encoding
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        label_encoder = LabelEncoder()
        df = df_raw.copy()
        df['차종_숫자'] = label_encoder.fit_transform(df['차종'])

        class_names = label_encoder.classes_  # ['일반', '전기차', '하이브리드'] 등

        # 특징/타겟 분리
        X = df[['배기량', '연비']]
        y = df['차종_숫자']

        # 3. 훈련/테스트 분리 및 스케일링
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import numpy as np

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. GridSearchCV로 최적 모델 학습 (캐싱으로 속도 향상)
        @st.cache_resource(show_spinner="모델 학습 중...")
        def train_best_model():
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

        best_model, best_params, best_cv_score = train_best_model()

        # 5. 모델 성능 표시
        st.subheader("📈 모델 성능")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("최적 CV 정확도", f"{best_cv_score:.4f}")
        with col2:
            y_pred = best_model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            st.metric("테스트 정확도", f"{test_acc:.4f}")
        with col3:
            st.metric("사용된 특징", "배기량, 연비")

        st.write("**최적 하이퍼파라미터**")
        st.json(best_params)

        # 6. Classification Report & Confusion Matrix
        st.subheader("🔍 상세 분류 보고서")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)

        st.subheader("🧩 혼동 행렬")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_xlabel('예측 차종')
        ax_cm.set_ylabel('실제 차종')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

        # 7. 사용자 입력으로 실시간 예측
        st.subheader("🔮 직접 예측해보기")
        col1, col2 = st.columns(2)
        with col1:
            displacement = st.number_input("배기량 (cc)", min_value=0, max_value=10000, value=2000, step=100)
        with col2:
            fuel_efficiency = st.number_input("연비 (km/L)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

        if st.button("차종 예측하기"):
            input_data = scaler.transform([[displacement, fuel_efficiency]])
            pred = best_model.predict(input_data)[0]
            pred_proba = best_model.predict_proba(input_data)[0]
            
            predicted_class = class_names[pred]
            proba_df = pd.DataFrame({
                '차종': class_names,
                '확률 (%)': np.round(pred_proba * 100, 2)
            }).sort_values(by='확률 (%)', ascending=False)

            st.success(f"### 예측 결과: **{predicted_class}**")
            st.dataframe(proba_df, use_container_width=True, hide_index=True)

            # 확률 바 차트
            fig_prob, ax_prob = plt.subplots(figsize=(8, 4))
            ax_prob.bar(proba_df['차종'], proba_df['확률 (%)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(class_names)])
            ax_prob.set_ylim(0, 100)
            ax_prob.set_ylabel('확률 (%)')
            ax_prob.set_title('각 차종별 예측 확률')
            for i, v in enumerate(proba_df['확률 (%)']):
                ax_prob.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')
            st.pyplot(fig_prob)

        st.info("""
        💡 **해석 팁**  
        • 전기차는 배기량이 0에 가까우며 연비가 매우 높음  
        • 하이브리드는 중간 정도의 배기량 + 높은 연비  
        • 일반 내연기관차는 배기량이 크고 연비가 상대적으로 낮음
        """)

    except Exception as e:
        st.error(f"전기차 분류 모델 페이지 오류: {str(e)}")
        st.info("'전기차분류.csv' 파일이 앱과 동일한 폴더에 있는지 확인해주세요.")
        
# ------------------- CCTV vs 사고 예측 모델 -------------------
elif menu == "CCTV vs 사고 예측 모델":
    st.header("📹 서울 자치구 CCTV vs 교통사고 분석 (2025)")

    try:
        # DB에서 데이터 불러오기
        query = """
        SELECT 
            year AS 연도,
            gu AS 자치구,
            cctv AS CCTV,
            accidents AS 사고건수
        FROM seoul_cctv_accident
        WHERE year = 2025
        ORDER BY gu
        """
        df = fetch_query(query)

        if df.empty:
            st.error("DB에서 데이터를 불러오지 못했습니다. 테이블(seoul_cctv_accident)을 확인하세요.")
            st.stop()

        # 데이터 준비
        X = df[['사고건수']].values  # 독립변수 (2D 배열 필요)
        y = df['CCTV'].values        # 종속변수

        # 훈련/테스트 데이터 분리 (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 선형회귀 모델 학습 (훈련 데이터만 사용)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 테스트 데이터로 성능 평가
        y_pred_test = model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        st.subheader("모델 성능 (훈련/테스트 분리)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² (테스트)", f"{r2_test:.4f}", "설명력")
        with col2:
            st.metric("MAE (테스트)", f"{mae_test:.1f}", "평균 절대 오차")
        with col3:
            st.metric("훈련 데이터 크기", f"{len(X_train)} / {len(X)}")

        st.write(f"회귀식: CCTV = {model.coef_[0]:.3f} × 사고건수 + {model.intercept_:.3f}")

        # 전체 데이터 테이블
        st.subheader("2025년 자치구별 데이터")
        st.dataframe(df[['자치구', 'CCTV', '사고건수']], use_container_width=True, hide_index=True)

        # 그래프: 전체 산점도 + 회귀선
        st.subheader("산점도 + 회귀선 (전체 데이터)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['사고건수'], df['CCTV'], color='darkorange', s=100, alpha=0.8, label='실제 데이터')
        x_range = np.linspace(df['사고건수'].min(), df['사고건수'].max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_range, color='blue', linewidth=3, label='선형회귀 모델')
        ax.set_title('서울 자치구별 사고건수 vs CCTV 개수 (2025년)', fontsize=14)
        ax.set_xlabel('사고건수')
        ax.set_ylabel('CCTV 개수')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 미래/가상 예측
        st.subheader("미래/가상 예측")
        col1, col2 = st.columns(2)
        with col1:
            accidents_input = st.number_input(
                "예상 사고건수 입력", min_value=0, max_value=5000, value=1500, step=100
            )
            predicted_cctv = int(round(model.predict([[accidents_input]])[0]))
            st.metric("예상 CCTV 개수", f"{predicted_cctv}대")

        st.info("""
        📊 **해석**   
        • 사고건수가 많을수록 CCTV도 많아지는 경향 (양의 상관).  
        • 이는 "사고 많은 곳에 CCTV를 우선 설치"한 정책 패턴으로 보입니다.
        """)

    except Exception as e:
        st.error(f"페이지 실행 중 오류: {str(e)}")
        st.info("DB 테이블(seoul_cctv_accident) 또는 쿼리를 확인해주세요.")