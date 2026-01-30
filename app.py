import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import altair as alt

# =====================================================
# PAGE CONFIG (⚠️ ЗААВАЛ ЭХНИЙ МӨРҮҮДИЙН НЭГ БАЙНА)
# =====================================================
st.set_page_config(
    page_title="Mongolbank Macro Data Dashboard",
    layout="wide"
)
# =====================================================
# HEADLINE CONFIG (EXTENSIBLE)
# =====================================================

HEADLINE_CONFIG = [
    {
        "type": "indicator",
        "code": "rgdp_2005",
        "title": "RGDP 2005",
        "subtitle": "Real GDP (2005 prices)",
        "topic": "gdp"
    },
    {
        "type": "indicator",
        "code": "rgdp_2010",
        "title": "RGDP 2010",
        "subtitle": "Real GDP (2010 prices)",
        "topic": "gdp"
    },
    {
        "type": "indicator",
        "code": "rgdp_2015",
        "title": "RGDP 2015",
        "subtitle": "Real GDP (2015 prices)",
        "topic": "gdp"
    },
    {
        "type": "indicator",
        "code": "ngdp",
        "title": "NGDP",
        "subtitle": "Nominal GDP",
        "topic": "gdp"
    },
    {
        "type": "indicator",
        "code": "growth",
        "title": "GDP Growth",
        "subtitle": "YoY growth",
        "topic": "gdp"
    },
    {
        "type": "population_total",   # 🔥 indicator_code биш!
        "title": "Population",
        "subtitle": "Total population",
        "topic": "population"
    }
]



# =====================================================
# APP START (TEST RENDER)
# =====================================================
st.title("🏦 Mongolbank Macro Data Dashboard")
st.caption("Quarterly GDP indicators (2000–2025)")
st.success("🔥 APP STARTED — UI rendering OK")
# =====================================================
# MAIN LAYOUT
# =====================================================
left_col, right_col = st.columns([1.4, 4.6], gap="large")
# =====================================================
# HEADLINE DATA LOADER (FILTER-INDEPENDENT)
# =====================================================
@st.cache_data(ttl=3600)
def load_headline_data():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(
        credentials=credentials,
        project=st.secrets["gcp_service_account"]["project_id"]
    )

    query = """
        SELECT
            year,
            indicator_code,
            value,
            topic,
            sex,
            age_group
        FROM `mongol-bank-macro-data.Automation_data.fact_macro`
        WHERE topic IN ('gdp', 'population')
        ORDER BY year
        """

    df = client.query(query).to_dataframe()

    # ✅ Python дээр canonical time үүсгэнэ
    df["year_num"] = df["year"].str[:4].astype(int)
    df["period"] = (
    df["year"]
    .str.extract(r"Q(\d)")
    .astype("Int64")
    )


    return df
# =====================================================
# KPI RENDER FUNCTIONS
# =====================================================

def render_gdp_kpi(df, gdp_type):

    if gdp_type == "GROWTH":
        series = df.groupby("time_label")["value"].mean()
    else:
        series = df.groupby("time_label")["value"].sum()

    if len(series) < 2:
        st.info("Not enough data for KPI calculation")
        return

    latest = series.iloc[-1]

    if gdp_type == "GROWTH":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest growth", f"{latest:.1f}%")
        k2.metric("Average", f"{series.mean():.1f}%")
        k3.metric("Peak", f"{series.max():.1f}%")
        k4.metric("Volatility", f"{series.std():.1f}")
    else:
        prev = series.iloc[-2]
        yoy = (latest / prev - 1) * 100

        k1, k2, k3 = st.columns(3)
        k1.metric("Latest level", f"{latest:,.0f}")
        k2.metric("Δ YoY", f"{yoy:.1f}%")
        k3.metric("Period average", f"{series.mean():,.0f}")


def render_pop_kpi(df):
    latest_period = df["time_label"].max()
    latest_df = df[df["time_label"] == latest_period]

    if latest_df.empty:
        st.info("No population data")
        return

    total = latest_df["value"].sum()

    gender = latest_df.groupby("sex")["value"].sum()
    male = gender.get("Эрэгтэй", gender.get("Male", 0))
    female = gender.get("Эмэгтэй", gender.get("Female", 0))
    age = latest_df.groupby("age_group")["value"].sum().sort_index()
    working_age = age.iloc[3:13].sum() if len(age) >= 13 else 0
    dependency = (total - working_age) / working_age * 100 if working_age else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total population", f"{total:,.0f}")
    k2.metric("Male", f"{male/total*100:.1f}%")
    k3.metric("Female", f"{female/total*100:.1f}%")
    k4.metric("Dependency ratio", f"{dependency:.1f}%")




# ================= LEFT COLUMN =================
with left_col:

    # ================= DATASET CARD =================
    with st.container(border=True):
        st.markdown("### 📦 Dataset")

        dataset = st.radio(
            "",
            ["GDP", "Population"],
            horizontal=True
        )

        # 1️⃣ topic ЭХЭЛЖ тодорхойлогдоно
        topic = dataset.lower()

    # 2️⃣ load_data FUNCTION (дуудахаас ӨМНӨ)
    @st.cache_data(ttl=3600)
    def load_data(topic):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(
            credentials=credentials,
            project=st.secrets["gcp_service_account"]["project_id"]
        )

        query = f"""
            SELECT
                year,
                indicator_code,
                value,
                sex,
                age_group,
                topic
            FROM `mongol-bank-macro-data.Automation_data.fact_macro`
            WHERE topic = '{topic}'
            ORDER BY year
        """
        df = client.query(query).to_dataframe()
        # ✅ TIME CANONICAL (ЭНД Л БҮГДИЙГ ШИЙДНЭ)
        if topic == "gdp":
            # GDP quarterly canonical time
            df["year_num"] = df["year"].str.split("-").str[0].astype(int)
            df["period"] = df["year"].str.split("-").str[1].astype("Int64")
            df["year"] = df["year_num"].astype(str)
            df["time_freq"] = "Q"
            df["sex"] = None
            df["age_group"] = None
        else:  # population
            df["year_num"] = df["year"].astype(int)
            df["period"] = None
            df["time_freq"] = "Y"
    
        return df

    # 3️⃣ DATA LOAD
    with st.spinner("⏳ Loading data from BigQuery..."):
        df = load_data(topic)



    # ---------- GDP TYPE SELECTOR ----------
    if topic == "gdp":
    
        with st.container(border=True):
            st.markdown("### 📊 GDP type")
            gdp_type = st.radio(
                "",
                ["RGDP2005", "RGDP2010", "RGDP2015", "NGDP", "GROWTH"],
                horizontal=True
            )
    
        prefix_map = {
            "RGDP2005": "rgdp_2005",
            "RGDP2010": "rgdp_2010",
            "RGDP2015": "rgdp_2015",
            "NGDP": "ngdp",
            "GROWTH": "growth"
        }
    
        prefix = prefix_map[gdp_type]
    
        available_indicators = sorted(
            df.loc[
                df["indicator_code"].str.contains(prefix, case=False, na=False),
                "indicator_code"
            ].unique()
        )
    
        # ✅ Indicators container GDP-ийн ДООР
        with st.container(border=True):
            st.markdown("### 📌 Indicators")
    
            selected_indicators = st.multiselect(
                "",
                available_indicators,
                default=available_indicators[:1] if available_indicators else []
            )
    
        filtered_df = df[df["indicator_code"].isin(selected_indicators)]
    
    # ---------- POPULATION ----------
    else:
        sex = st.multiselect(
            "Sex",
            sorted(df["sex"].dropna().unique()),
            default=[]
        )
    
        age_group = st.multiselect(
            "Age group",
            sorted(df["age_group"].dropna().unique()),
            default=[]
        )
    
        if sex and age_group:
            filtered_df = df[
                df["sex"].isin(sex) &
                df["age_group"].isin(age_group)
            ]
        else:
            filtered_df = df.copy()
    #=====================================
    # ⏱ Frequency (GLOBAL, SINGLE)
    #=====================================
    with st.container(border=True):
        st.markdown("### ⏱ Frequency")
    
        freq = st.radio(
            "",
            ["Yearly", "Quarterly", "Monthly"],
            horizontal=True
        )
    
    freq_map = {
        "Yearly": "Y",
        "Quarterly": "Q",
        "Monthly": "M"
    }
    selected_freq = freq_map[freq]
    
    # ==============================
    # 🔎 DATA-AWARE FILTER (SAFE)
    # ==============================
    if filtered_df.empty:
        st.warning("⚠️ No data available for selected filters.")
        time_filtered_df = pd.DataFrame()
    
    elif selected_freq not in filtered_df["time_freq"].unique():
        st.warning(
            f"⚠️ This dataset does not contain {freq.lower()} data."
        )
        time_filtered_df = pd.DataFrame()
    
    else:
        time_filtered_df = (
            filtered_df
            .loc[filtered_df["time_freq"] == selected_freq]
            .copy()
        )
    # ⏳ TIME RANGE (FREQUENCY-AWARE)
    # ==============================
    if not time_filtered_df.empty:
    
        with st.container(border=True):
            st.markdown("### ⏳ Time range")
    
            if selected_freq == "Y":
                year_list = sorted(time_filtered_df["year"].unique())
            
                start_y = st.selectbox(
                    "Start year",
                    year_list,
                    index=0
                )
            
                end_y = st.selectbox(
                    "End year",
                    year_list,
                    index=len(year_list) - 1
                )
            
                # 🔒 SAFETY CHECK
                if start_y > end_y:
                    st.error("❌ Start year must be before End year")
                    time_filtered_df = pd.DataFrame()
                else:
                    time_filtered_df = time_filtered_df[
                        (time_filtered_df["year"] >= start_y) &
                        (time_filtered_df["year"] <= end_y)
                    ]

            elif selected_freq == "Q":
                time_filtered_df["t"] = (
                    time_filtered_df["year"].astype(str)
                    + "-Q"
                    + time_filtered_df["period"].astype(str)
                )
            
                t_list = sorted(time_filtered_df["t"].unique())
            
                start_q = st.selectbox("Start quarter", t_list, index=0)
                end_q = st.selectbox("End quarter", t_list, index=len(t_list)-1)
            
                if start_q > end_q:
                    st.error("❌ Start quarter must be before End quarter")
                    time_filtered_df = pd.DataFrame()
                else:
                    time_filtered_df = time_filtered_df[
                        (time_filtered_df["t"] >= start_q) &
                        (time_filtered_df["t"] <= end_q)
                    ]

            elif selected_freq == "M":
                # 🧱 Temporary time key
                time_filtered_df["t"] = (
                    time_filtered_df["year"].astype(str)
                    + "-"
                    + time_filtered_df["period"].astype(str).str.zfill(2)
                )
            
                t_list = sorted(time_filtered_df["t"].unique())
            
                start_m = st.selectbox("Start month", t_list, index=0)
                end_m = st.selectbox("End month", t_list, index=len(t_list) - 1)
            
                # 🔒 SAFETY CHECK
                if start_m > end_m:
                    st.error("❌ Start month must be before End month")
                    time_filtered_df = pd.DataFrame()
                else:
                    time_filtered_df = time_filtered_df[
                        (time_filtered_df["t"] >= start_m) &
                        (time_filtered_df["t"] <= end_m)
                    ]
                                # 🧹 REMOVE TEMP COLUMN
    time_filtered_df = time_filtered_df.drop(columns=["t"], errors="ignore")
    
    # =============================
    # CREATE TIME LABEL (STANDARD)
    # =============================
    if not time_filtered_df.empty:
    
        if selected_freq == "Y":
            time_filtered_df["time_label"] = time_filtered_df["year"].astype(str)
    
        elif selected_freq == "Q":
            time_filtered_df["time_label"] = (
                time_filtered_df["year"].astype(str)
                + "-Q"
                + time_filtered_df["period"].astype(str)
            )
    
        elif selected_freq == "M":
            time_filtered_df["time_label"] = (
                time_filtered_df["year"].astype(str)
                + "-"
                + time_filtered_df["period"].astype(str).str.zfill(2)
            )

    # ---------- SERIES COLUMN (POPULATION) ----------
    if topic == "population":
        time_filtered_df = time_filtered_df.copy()
        time_filtered_df["Series"] = (
            time_filtered_df["sex"].astype(str)
            + " | "
            + time_filtered_df["age_group"].astype(str)
        )
    # ================= RIGHT COLUMN =================
    with right_col:
        with st.container(border=True):
            st.markdown("### 📈 Main chart")
        
            if time_filtered_df.empty:
                st.warning("No data for selected filters")
    
            else:
                # ===== GDP (HEADLINE ONLY) =====
                if topic == "gdp":
                
                    plot_df = time_filtered_df.copy()
                
                    # =============================
                    # 🔒 DEFENSIVE CHECK — ЯГ ЭНД
                    # =============================
                    if "time_label" not in plot_df.columns:
                        st.error("❌ time_label is missing — check frequency logic")
                
                    else:
                        chart_df = (
                            plot_df
                            .pivot_table(
                                index="time_label",
                                columns="indicator_code",
                                values="value",
                                aggfunc="mean"
                            )
                        )
                
                        st.line_chart(chart_df)

                # ===== POPULATION =====
                else:
                    plot_df = time_filtered_df.copy()
                
                    chart = (
                        alt.Chart(plot_df)
                        .mark_line()
                        .encode(
                            x=alt.X("time_label:N", title="Time"),
                            y=alt.Y("value:Q", title="Population"),
                            color=alt.Color(
                                "Series:N",
                                legend=alt.Legend(orient="right")
                            ),
                            tooltip=["time_label", "sex", "age_group", "value"]
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)

          # ===== DOWNLOAD OVERLAY (BOTTOM-RIGHT, ULTRA MINIMAL) =====
            st.markdown(
                """
                <style>
                /* Parent container must be relative */
                div[data-testid="stVerticalBlock"] {
                    position: relative;
                }
            
                /* Target Streamlit download button wrapper */
                div[data-testid="stDownloadButton"] {
                    position: absolute;
                    bottom: 14px;
                    left: 14px;          /* ⬅️ ЭСРЭГ ТАЛ = БАРУУН ДООД */
                    z-index: 10;
                }
            
                /* Actual button styling */
                div[data-testid="stDownloadButton"] button {
                    background-color: rgba(30, 41, 59, 0.4);  /* background-тай бараг адил */
                    color: rgba(203, 213, 225, 0.7);
                    border: none;         /* ⛔ ХҮРЭЭ БҮРЭН АЛГА */
                    padding: 3px 5px;
                    font-size: 11px;
                    border-radius: 4px;
                    line-height: 1;
                    box-shadow: none;
                    cursor: pointer;
                }
            
                div[data-testid="stDownloadButton"] button:hover {
                    background-color: rgba(30, 41, 59, 0.7);
                    color: rgba(248, 250, 252, 0.95);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.download_button(
                label="↓",
                data=plot_df.to_csv(index=False),
                file_name="main_chart_data.csv",
                mime="text/csv",
                help="Download chart data",
                key="main_chart_download"
            )
        # =====================================================
        # KPI CONTAINER (BELOW MAIN CHART)
        # =====================================================
        if not time_filtered_df.empty:
        
            with st.container(border=True):
                st.markdown("### 📌 Key indicators")
    
                if topic == "gdp":
                    render_gdp_kpi(time_filtered_df, gdp_type)
    
                elif topic == "population":
                    render_pop_kpi(time_filtered_df)
        # =====================================================
        # TOP 4 SECTOR BREAKDOWN — GDP ONLY
        # =====================================================
        
        if (
            topic == "gdp"
            and gdp_type == "GROWTH"
            and not time_filtered_df.empty
        ):
        
            df_nz = time_filtered_df.copy()
            df_nz["value"] = pd.to_numeric(df_nz["value"], errors="coerce")
        
            df_nz = df_nz[
                df_nz["indicator_code"].str.startswith("growth_")
                & df_nz["value"].notna()
                & (df_nz["value"] != 0)
            ]
        
            if df_nz.empty:
                st.info("No non-zero GDP growth data.")
            else:
                latest_row = (
                    df_nz
                    .sort_values(["year_num", "period"])
                    .iloc[-1]
                )
        
                latest_label = f"{latest_row['year_num']}-Q{int(latest_row['period'])}"
        
                sector_df = (
                    time_filtered_df[
                        (time_filtered_df["time_label"] == latest_label)
                        & time_filtered_df["indicator_code"].str.startswith("growth_")
                    ][["indicator_code", "value"]]
                    .copy()
                )
        
                sector_df["value"] = pd.to_numeric(sector_df["value"], errors="coerce")
                sector_df = sector_df.dropna()
                sector_df = sector_df[sector_df["value"] != 0]
        
                sector_df["sector"] = (
                    sector_df["indicator_code"]
                    .str.replace("growth_", "", regex=False)
                    .str.replace("_", " ")
                    .str.title()
                )
        
                top_pos = sector_df.sort_values("value", ascending=False).head(2)
                top_neg = sector_df.sort_values("value").head(2)
        
                st.markdown(f"### 📊 Sector contribution ({latest_label})")
        
                c1, c2 = st.columns(2)
        
                with c1:
                    st.markdown("#### 🚀 Highest growth")
                    for _, r in top_pos.iterrows():
                        st.metric(r["sector"], f"{r['value']:.1f} pp")
        
                with c2:
                    st.markdown("#### ⚠️ Largest decline")
                    for _, r in top_neg.iterrows():
                        st.metric(r["sector"], f"{r['value']:.1f} pp")
        # =====================================================
        # TOP 4 POPULATION STRUCTURE
        # =====================================================
        elif topic == "population" and not time_filtered_df.empty:
        
            latest_label = time_filtered_df["time_label"].max()
        
            pop_df = (
                time_filtered_df[
                    time_filtered_df["time_label"] == latest_label
                ]
                .groupby("Series")["value"]
                .sum()
                .reset_index()
                .sort_values("value", ascending=False)
                .head(4)
            )
        
            st.markdown(f"### 👥 Population structure ({latest_label})")
        
            for _, r in pop_df.iterrows():
                st.metric(r["Series"], f"{r['value']:,.0f}")

















# =====================================================
# HEADLINE INDICATORS (EXTENSIBLE, FILTER-INDEPENDENT)
# =====================================================

headline_df = load_headline_data()

st.markdown("## 📊 Headline indicators")

N_COLS = 4
rows = [
    HEADLINE_CONFIG[i:i + N_COLS]
    for i in range(0, len(HEADLINE_CONFIG), N_COLS)
]

for row in rows:
    cols = st.columns(N_COLS)
    for col, cfg in zip(cols, row):
        with col:
            with st.container(border=True):
    
                st.markdown(f"""
                <div style="text-align:center">
                    <div style="font-weight:600">{cfg["title"]}</div>
                    <div style="font-size:12px;opacity:0.6">{cfg["subtitle"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # ===== GDP INDICATOR (SMART HEADLINE) =====
                if cfg["type"] == "indicator":
                
                    base_df = headline_df[
                        headline_df["indicator_code"]
                        .fillna("")
                        .str.lower()
                        .str.startswith(cfg["code"])
                    ].copy()
                
                    # 🔥 RGDP / NGDP → yearly SUM (4 улирал)
                    if cfg["code"] in ["rgdp_2005", "rgdp_2010", "rgdp_2015", "ngdp"]:
                        plot_df = (
                            base_df
                            .groupby("year", as_index=True)["value"]
                            .sum()
                            .to_frame()
                            .sort_index()
                        )
                
                    # 🔥 GROWTH → yearly MEAN
                    elif cfg["code"] == "growth":
                        plot_df = (
                            base_df
                            .groupby("year_num", as_index=True)["value"]
                            .mean()
                            .to_frame()
                            .sort_index()
                        )
                
                    # fallback (аюулгүй)
                    else:
                        plot_df = (
                            base_df
                            .drop_duplicates(subset=["year_num"])
                            .set_index("year_num")[["value"]]
                            .sort_index()
                        )


                # ===== POPULATION TOTAL =====
                elif cfg["type"] == "population_total":
                    plot_df = (
                        headline_df[
                            (headline_df["topic"] == "population") &
                            (headline_df["sex"] == "Бүгд") &
                            (headline_df["age_group"] == "Бүгд")
                        ]
                        .set_index("year_num")[["value"]]
                        .sort_index()
                    )

                st.line_chart(plot_df, height=160)
                
# =====================================================
# RAW DATA Preview
# =====================================================
with st.expander("📄 Raw data"):

    # ===================== GDP =====================
    if topic == "gdp":
    
        raw_prefix = prefix_map[gdp_type]
    
        # ✅ ЯГ ЭНД — FILTER ОРООГҮЙ ЭХ ӨГӨГДӨЛ
        raw_df = df.copy()
        
        # ✅ CANONICAL TIME_LABEL (RAW-д ЗААВАЛ НЭГ УДАА)
        raw_df["time_label"] = (
            raw_df["year"]
            + "-Q"
            + raw_df["period"].astype(str)
        )
        df_pivot = (
            raw_df
            .pivot_table(
                index="time_label",
                columns="indicator_code",
                values="value",
                aggfunc="mean"
            )
            .reset_index()
        )
        ordered_cols = (
            ["time_label"] +
            sorted(c for c in df_pivot.columns if c.startswith(raw_prefix))
        )
    
        st.dataframe(df_pivot[ordered_cols], use_container_width=True)


    # ===================== POPULATION =====================
    else:
        raw_df = df.copy()
    
        df_pivot = (
            raw_df
            .pivot_table(
                index=["sex", "age_group"],
                columns="year",
                values="value",
                aggfunc="sum"
            )
            .reset_index()
        )
    
        st.dataframe(df_pivot, use_container_width=True)
