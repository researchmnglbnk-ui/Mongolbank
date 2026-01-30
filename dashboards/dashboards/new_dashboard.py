import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

# ==================
# PAGE
# ==================
st.set_page_config("Dashboard", layout="wide")
st.title("🏦 Dashboard")
st.caption("Macro Indicators")

# ✅ GLOBAL STYLE (END USER QUALITY)
st.markdown("""
<style>
/* Page width control */
.block-container {
    padding-top: 3.2rem;
    padding-bottom: 2.2rem;
}
h1 {
    margin-top:0;
}
/* Sidebar-like left column feel */
div[data-testid="column"]:first-child {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
}

/* Section headers */
h2, h3 {
    letter-spacing: 0.3px;
}

/* Remove Altair gray background */
.vega-embed {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parents[1]
EXCEL_PATH = BASE_DIR / "Dashboard_cleaned_data.xlsx"

@st.cache_data(show_spinner=False)
def read_sheet(sheet):
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=[0, 1])
# ======================
# 🔑 PERCENT INDICATOR KEYWORDS
# ======================
percentage_keywords = [
    "Taylor",
    "Hodrick-Prescott",
    "Beveridge Nelson",
    "Band-Pass",
    "Policy rate",
    "Neutral rate",
    "Inflation",
    "Forecast 1",
    "Forecast 2",
    "ECPI",
    "UB inflation",
    "Hodrick-Prescott",
    "Kalman",
    "Production function",
    "Average",
    "GDP, Yoy",
    "Dynamic Factor Model",
    "Deviation",
    "Household loan",
    "Corporate loan",
    "Household loan supply",
    "Corporate loan supply"
]

def is_percentage_indicator(name: str) -> bool:
    name_l = name.lower()
    return any(k in name_l for k in percentage_keywords)


# =====================
# DATASET SELECT
# ======================
sheets = [s for s in pd.ExcelFile(EXCEL_PATH).sheet_names
          if s.lower() in ["month", "quarter"]]

left, right = st.columns([1.4, 4.6], gap="large")

with left:
    with st.container(border=True):
        st.subheader("📦 Dataset")

        dataset = st.radio(
            "Dataset",
            sheets,
            horizontal=True,
            label_visibility="collapsed"
        )
        
# ======================
# LOAD DATA
# ======================
df = read_sheet(dataset)

# ======================
# HEADER-ийг ШИНЭЧЛЭХ
# ======================
# Excel-ийн бүтцийг хадгална
if isinstance(df.columns, pd.MultiIndex):
    # Зөвхөн эхний түвшний header-ыг шалгана
    top_level = df.columns.get_level_values(0)
    
    # TIME багануудыг олох
    time_cols = []
    for col in df.columns:
        if col[0] in ["Year", "Month", "Quarter"]:
            time_cols.append(col)
    
    if not time_cols:
        st.error("❌ No time columns found")
        st.stop()
    
    # TIME ба DATA салгах
    df_time = df[time_cols].copy()
    df_data = df.drop(columns=time_cols)
    
    freq = "Monthly" if "Month" in df_time.columns else "Quarterly"
    
    with left:
        st.caption(f"Frequency: {freq}")
        
    # TIME багануудыг хялбарчилна
    for i, col in enumerate(df_time.columns):
        if isinstance(col, tuple):
            df_time.columns.values[i] = col[0]  # Зөвхөн эхний түвшний нэрийг ашиглана
    
    # DATA-ийн header-ыг цэвэрлэх
    # Level 0-ийг цэвэрлэх (Unnamed устгах)
    level0 = df_data.columns.get_level_values(0)
    level1 = df_data.columns.get_level_values(1)
    
    # Level 0-д байгаа "Unnamed" утгуудыг өмнөх утгаар дүүргэх
    new_level0 = []
    for val in level0:
        if pd.isna(val) or "Unnamed" in str(val):
            new_level0.append(new_level0[-1] if new_level0 else "Other")
        else:
            new_level0.append(val)
    
    df_data.columns = pd.MultiIndex.from_arrays([new_level0, level1])
    
else:
    # Хэрэв MultiIndex биш бол (баталгаажуулалт)
    st.error("❌ Unexpected data format - expected MultiIndex columns")
    st.stop()
    
with left:
    # ======================
    # 🧭 INDICATOR GROUP (ТУСДАА ХҮРЭЭ)
    # ======================
    with st.container(border=True):
        st.subheader("🧭 Indicator group")

        available_groups = sorted(df_data.columns.get_level_values(0).unique())
        group = st.radio(
            "Indicator group",
            available_groups,
            label_visibility="collapsed"
        )
    # ======================
    # 📌 INDICATORS (ТУСДАА ХҮРЭЭ)
    # ======================
    with st.container(border=True):
        st.subheader("📌 Indicators")

        indicators = sorted([
            col[1] for col in df_data.columns
            if col[0] == group and not pd.isna(col[1])
        ])

        selected = st.multiselect(
            "Indicators",
            indicators,
            default=[indicators[0]] if indicators else [],
            label_visibility="collapsed"
        )

# ======================
# DATA PREPARATION
# ======================
if not selected:
    st.info("ℹ️ No indicators selected — showing group-level summary only.")

# ======================
# 🔧 KPI & CHANGE HELPERS (GLOBAL)
# ======================
def compute_changes(df, indicator, freq):
    s = df[["x", indicator]].dropna().copy()

    # 🔒 X хамгаалалт (ЧИНИЙ ХҮССЭН ХЭСЭГ)
    s["x"] = s["x"].astype(str).str.strip()
    s = s[s["x"] != ""]

    if len(s) < 2:
        return None

    # 🔒 SORT
    s = s.sort_values("x").reset_index(drop=True)

    # 🔒 VALUE SCALAR
    latest_val = float(s.iloc[-1][indicator])
    prev_val   = float(s.iloc[-2][indicator])
    
    # 🔍 Percentage индикатор мөн эсэхийг шалгах
    is_percentage = is_percentage_indicator(indicator)

    # ======================
    # 🔹 PREV (QoQ / MoM) - PERCENTAGE-ийн хувьд ялгаатай тооцоо
    # ======================
    if is_percentage:
        # Percentage утгын хувьд: хамгийн сүүлийн утга - өмнөх утга
        prev = (latest_val - prev_val) if prev_val is not None else None
    else:
        # Бусад утгын хувьд: хувиар тооцоолно
        prev = (latest_val / prev_val - 1) * 100 if prev_val != 0 else None

    # ======================
    # 🔹 YoY (INDEX-BASED) - PERCENTAGE-ийн хувьд ялгаатай тооцоо
    # ======================
    yoy = None
    if freq == "Quarterly" and len(s) >= 5:
        base_val = float(s.iloc[-5][indicator])
        if base_val is not None:
            if is_percentage:
                # Percentage утгын хувьд: хамгийн сүүлийн утга - жилийн өмнөх утга
                yoy = (latest_val - base_val)
            else:
                # Бусад утгын хувьд: хувиар тооцоолно
                yoy = (latest_val / base_val - 1) * 100 if base_val != 0 else None

    elif freq == "Monthly" and len(s) >= 13:
        base_val = float(s.iloc[-13][indicator])
        if base_val is not None:
            if is_percentage:
                # Percentage утгын хувьд: хамгийн сүүлийн утга - жилийн өмнөх утга
                yoy = (latest_val - base_val)
            else:
                # Бусад утгын хувьд: хувиар тооцоолно
                yoy = (latest_val / base_val - 1) * 100 if base_val != 0 else None

    # ======================
    # 🔹 YTD - PERCENTAGE-ийн хувьд ялгаатай тооцоо
    # ======================
    ytd = None
    try:
        current_year = s.iloc[-1]["x"][:4]
        year_data = s[s["x"].str.startswith(current_year)]
        if len(year_data) >= 1:
            year_start = float(year_data.iloc[0][indicator])
            if year_start is not None:
                if is_percentage:
                    # Percentage утгын хувьд: хамгийн сүүлийн утга - жилийн эхний утга
                    ytd = (latest_val - year_start)
                else:
                    # Бусад утгын хувьд: хувиар тооцоолно
                    ytd = (latest_val / year_start - 1) * 100 if year_start != 0 else None
    except:
        ytd = None

    return {
        "latest": latest_val,
        "prev": prev,
        "yoy": yoy,
        "ytd": ytd
    }

def render_change(label, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"<span class='change-item'>{label}: N/A</span>"

    arrow = "▲" if value > 0 else "▼"
    cls = "change-up" if value > 0 else "change-down"

    return (
        f"<span class='change-item {cls}'>"
        f"<span class='change-arrow'>{arrow}</span>"
        f"{label}: {value:.2f}%"
        f"</span>"
    )


# Өгөгдлийг цуваа болгон нэгтгэх
series = df_time.copy()
# ======================
# HELPER: DataFrame → Series болгох
# ======================
def as_series(col):
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col

# ======================
# FIX: Year / Month / Quarter block structure
# ======================
for col in ["Year", "Month", "Quarter"]:
    if col in series.columns:
        series[col] = series[col].ffill()

# Time багануудыг тоон утга болгох
for col in ["Year", "Month", "Quarter"]:
    if col in series.columns:
        # Баганын утгуудыг list болгон авах, дараа нь Series болгох
        values = series[col].values.tolist() if hasattr(series[col], 'values') else series[col]
        # Хэрэв nested list байвал задлах
        if isinstance(values, list) and values and isinstance(values[0], list):
            values = [v[0] if isinstance(v, list) else v for v in values]
        series[col] = pd.to_numeric(pd.Series(values), errors='coerce')
# ======================
# CREATE TIME INDEX (FINAL, SAFE)
# ======================
year = as_series(series["Year"]) if "Year" in series.columns else None
month = as_series(series["Month"]) if "Month" in series.columns else None
quarter = as_series(series["Quarter"]) if "Quarter" in series.columns else None

if year is not None and month is not None:
    series["time"] = (
        year.astype(int).astype(str) + "-" +
        month.astype(int).astype(str).str.zfill(2)
    )

elif year is not None and quarter is not None:
    series["time"] = (
        year.astype(int).astype(str) + "-Q" +
        quarter.astype(int).astype(str)
    )

elif year is not None:
    series["time"] = year.astype(int).astype(str)

else:
    st.error("❌ No valid time columns found")
    st.stop()
# ======================
# ✅ YEAR LABEL (GLOBAL X AXIS)
# ======================
series["year_label"] = series["Year"].astype(int).astype(str)

for col in ["Year", "Month", "Quarter"]:
    if col in series.columns:
        series[col] = as_series(series[col])

# ======================
# ⏳ TIME RANGE (MAIN CHART ONLY)
# ======================
with left:
    with st.container(border=True):
        st.subheader("⏳ Time range")
        
        # Жилийн сонголтыг хоёр баганад зэрэгцүүлэх
        year_col1, year_col2 = st.columns(2)
        
        year_s = series["Year"]
        if isinstance(year_s, pd.DataFrame):
            year_s = year_s.iloc[:, 0]
        
        years = sorted(
            year_s.dropna().astype(int).unique().tolist()
        )
        
        with year_col1:
            start_year = st.selectbox(
                "Start Year",
                years,
                index=0
            )
        
        with year_col2:
            end_year = st.selectbox(
                "End Year",
                years,
                index=len(years)-1
            )
        
        # Сар эсвэл улирлын сонголтыг хоёр баганад зэрэгцүүлэх
        if freq == "Monthly":
            months = list(range(1, 13))
            
            month_col1, month_col2 = st.columns(2)
            
            with month_col1:
                start_month = st.selectbox(
                    "Start Month",
                    months,
                    index=0,
                    format_func=lambda x: f"{x:02d}"
                )
            
            with month_col2:
                end_month = st.selectbox(
                    "End Month",
                    months,
                    index=len(months)-1,
                    format_func=lambda x: f"{x:02d}"
                )
            
            # time string үүсгэх
            start_time = f"{start_year}-{start_month:02d}"
            end_time = f"{end_year}-{end_month:02d}"
            
        elif freq == "Quarterly":
            quarters = [1, 2, 3, 4]
            
            quarter_col1, quarter_col2 = st.columns(2)
            
            with quarter_col1:
                start_quarter = st.selectbox(
                    "Start Quarter",
                    quarters,
                    index=0
                )
            
            with quarter_col2:
                end_quarter = st.selectbox(
                    "End Quarter",
                    quarters,
                    index=len(quarters)-1
                )
            
            # time string үүсгэх
            start_time = f"{start_year}-Q{start_quarter}"
            end_time = f"{end_year}-Q{end_quarter}"

# Сонгосон үзүүлэлтүүдийг нэмэх
for indicator in selected:
    if (group, indicator) in df_data.columns:
        series[indicator] = df_data[(group, indicator)].values
    else:
        st.warning(f"Indicator '{indicator}' not found in data")

# Графикийн өгөгдөл бэлтгэх
plot_data = (
    series
    .loc[:, ["time"] + selected]
    .copy()
    .set_index("time")
    .sort_index()
)
# ======================
# SPLIT: DATA vs NO DATA
# ======================

# өгөгдөлтэй баганууд
valid_cols = [
    col for col in plot_data.columns
    if not plot_data[col].isna().all()
]

# өгөгдөлгүй баганууд
nodata_cols = [
    col for col in plot_data.columns
    if plot_data[col].isna().all()
]

# зөвхөн өгөгдөлтэйг графикт ашиглана
plot_data_valid = plot_data[valid_cols]
# ======================
# 🔒 HARD CHECK: time column
# ======================
if "time" not in series.columns:
    st.error("❌ 'time' column was not created. Check Year / Month / Quarter logic.")
    st.stop()

# time хоосон эсэх
if series["time"].isna().all():
    st.error("❌ 'time' column exists but contains only NaN")
    st.stop()
        

# ======================
# MAIN CHART (PRO-LEVEL: ZOOM + PAN + SCROLL)
# ======================
with right:
    with st.container(border=True):
        
        # ===== 1️⃣ DATA (NO AGGREGATION)
        chart_df = series[["time"] + selected].copy()
        
        # ⏳ APPLY TIME RANGE (SAFE STRING FILTER)
        chart_df = chart_df[
            (chart_df["time"] >= start_time) & 
            (chart_df["time"] <= end_time)
        ]
        
        # ===== 2️⃣ VALID INDICATORS ONLY
        valid_indicators = [
            c for c in selected
            if c in chart_df.columns and not chart_df[c].isna().all()
        ]
        
        if not valid_indicators:
            st.warning("⚠️ No data available for selected indicator(s)")
            st.stop()

        import altair as alt
        
        # ===== 3️⃣ TIME FORMATTING =====
        chart_df = chart_df.copy()
        
        if freq == "Monthly":
            chart_df["time_dt"] = pd.to_datetime(
                chart_df["time"],
                format="%Y-%m",
                errors="coerce"
            )
        elif freq == "Quarterly":
            chart_df["time_dt"] = (
                pd.PeriodIndex(chart_df["time"], freq="Q")
                .to_timestamp()
            )
        else:
            st.error("❌ Unknown frequency")
            st.stop()
        
        # 🔒 HARD CHECK
        if chart_df["time_dt"].isna().all():
            st.error("❌ Failed to convert time → datetime")
            st.stop()
        # 🔥 FIX: START MAIN CHART FROM FIRST REAL DATA POINT
        first_valid_time = chart_df.loc[
            chart_df[valid_indicators].notna().any(axis=1),
            "time_dt"
        ].min()

        chart_df = chart_df[chart_df["time_dt"] >= first_valid_time]

        # ===== 4️⃣ X-AXIS CONFIGURATION =====
        # Жилийн тооцоо
        start_year_int = int(start_year) if isinstance(start_year, str) else start_year
        end_year_int = int(end_year) if isinstance(end_year, str) else end_year
        year_count = end_year_int - start_year_int + 1
        
        # ✅ ЯГ ӨМНӨХ ШИГЭЭ: 2 ЖИЛИЙН ИНТЕРВАЛТАЙ ШОШГО
        # Хэрэв year_count 12-оос их бол 2 жил тутамд, бага бол жил бүр
        if year_count > 12:
            tick_step = 2
        else:
            tick_step = 1
        
        # X тэнхлэгийн тохируулга - ЯГ ӨМНӨХ ШИГ
        x_axis = alt.Axis(
            title=None,
            labelAngle=0,
            labelFontSize=11,
            grid=False,
            domain=True,
            orient='bottom',
        
            labelExpr="""
            timeFormat(
              datum.value,
              (timeOffset('month', datum.value, 1) - datum.value) < 1000*60*60*24*40
                ? '%Y-%m'
                : '%Y'
            )
            """
        )

        # ===== 5️⃣ LEGEND ТОХИРУУЛГА - ЯГ ӨМНӨХ ШИГЭЭ БАРУУН ТАЛД =====
        legend_config = alt.Legend(
            title=None,
            orient='right',  # ✅ ЯГ ӨМНӨХ ШИГЭЭ БАРУУН ТАЛД
            offset=0,
            padding=0,
            labelFontSize=11,
            symbolType="stroke",
            symbolSize=80,
            direction='vertical',
            # ✅ ЯГ ӨМНӨХ ШИГЭЭ ДЭВСГЭРГҮЙ, ЦЭВЭР
            fillColor=None,
            strokeColor=None,
            cornerRadius=0,
            labelLimit=180
        )
        # ===== 6️⃣ SHARED BRUSH/ZOOM SELECTION =====
        # ЗӨВЛӨГӨӨ: НЭГ selection_interval ашиглан хоёр графикийг холбоно
        zoom_brush = alt.selection_interval(
            encodings=['x'],
            bind='scales',  # Mouse wheel zoom + drag pan
            translate=True,  # Зүүн баруун тийш гүйлгэх
            zoom=True,       # Zoom идэвхжүүлэх
            empty=False      # Анхны байдлаар бүх өгөгдөл харагдана
        )
        # ===== 1️⃣1️⃣ MINI OVERVIEW - ЯГ ӨМНӨХ ШИГЭЭ ХЭМЖЭЭ =====
        # MINI CHART-д ЗӨВХӨН PAN (NO ZOOM) - FRED ШИГЭЭ
        mini_brush = alt.selection_interval(
            encodings=['x'],
            translate=True,   # Зүүн баруун тийш гүйлгэх
            zoom=False,       # ❌ ZOOM ХИЙХГҮЙ
            empty=False
        )
        
        # ===== 7️⃣ BASE CHART - ЯГ ӨМНӨХ ШИГЭЭ =====
        base = (
            alt.Chart(chart_df)
            .transform_fold(
                valid_indicators,
                as_=["Indicator", "RawValue"]
            )
            .transform_calculate(
                DisplayValue="""
                datum.RawValue == null || isNaN(datum.RawValue)
                ? null
                : (
                    indexof(
                        %s,
                        lower(datum.Indicator)
                    ) >= 0
                    ? datum.RawValue * 100
                    : datum.RawValue
                  )
                """ % str([k.lower() for k in percentage_keywords])
            )

            .encode(
                x=alt.X(
                    "time_dt:T",
                    title=None,
                    axis=x_axis,
                    scale=alt.Scale(zero=False, domain=mini_brush)
                ),
                y=alt.Y(
                    "DisplayValue:Q",
                    title=None,
                    axis=alt.Axis(
                        grid=True,
                        gridOpacity=0.25,
                        domain=True,
                        labelFontSize=11,
                        offset=5,
                        format=",.2f"
                    )
                ),
                color=alt.Color(
                    "Indicator:N",
                    legend=legend_config
                ),
                tooltip=[
                    alt.Tooltip(
                        "time_dt:T",
                        title="Time",
                        format="%Y-%m" if freq == "Monthly" else "%Y-Q%q"
                    ),
                    alt.Tooltip("Indicator:N"),
                    alt.Tooltip("DisplayValue:Q", format=",.2f", title="Value")
                ]
            )
        )

        
        # ===== 8️⃣ HOVER СОНГОЛТ - ЯГ ӨМНӨХ ШИГ =====
        hover = alt.selection_single(
            fields=["time_dt"],
            nearest=True,
            on="mouseover",
            empty=False,
            clear="mouseout"
        )
        
        # 🔥 CREDIT SUPPLY DETECTION
        is_credit_supply = (group == "Credit supply" and freq == "Quarterly")
        
        household_bar = None
        corporate_bar = None
        household_line = None
        corporate_line = None
        
        if is_credit_supply:
            household_bar = next((ind for ind in valid_indicators if "issued" in ind.lower() and "household" in ind.lower()), None)
            corporate_bar = next((ind for ind in valid_indicators if "issued" in ind.lower() and "corporate" in ind.lower()), None)
            household_line = next((ind for ind in valid_indicators if "household" in ind.lower() and "supply" in ind.lower() and "issued" not in ind.lower()), None)
            corporate_line = next((ind for ind in valid_indicators if "corporate" in ind.lower() and "supply" in ind.lower() and "issued" not in ind.lower()), None)
        # ===== 9️⃣ ГРАФИК ЭЛЕМЕНТҮҮД - ЯГ ӨМНӨХ ШИГ =====
        line = (
            base
            .transform_filter(alt.datum.DisplayValue != None)
            .mark_line(strokeWidth=2.4)
            .encode(
                y=alt.Y(
                    "DisplayValue:Q",
                    title=None,
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        grid=True,
                        gridOpacity=0.25,
                        domain=True,
                        labelFontSize=11,
                        offset=5,
                        format=",.2f"
                    )
                )
            )
        )

        points = (
            base
            .mark_circle(
                size=65,  # ✅ ЯГ ӨМНӨХ ШИГ (65)
                filled=True,
                stroke="#ffffff",
                strokeWidth=2  # ✅ ЯГ ӨМНӨХ ШИГ
            )
            .encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            )
            .add_params(hover)
        )

        # ===== 🔴 LAST VALUE MARKER (MAIN CHART ONLY) =====
        last_point = (
            base
            # 🔑 1. NULL утгуудыг бүрэн хасна
            .transform_filter(
                alt.datum.RawValue != None
            )
            # 🔑 2. Indicator бүрийн хамгийн сүүлийн бодит огноог олно
            .transform_window(
                rank="rank(time_dt)",
                sort=[alt.SortField("time_dt", order="descending")],
                groupby=["Indicator"]
            )
            # 🔑 3. Зөвхөн rank == 1
            .transform_filter(
                alt.datum.rank == 1
            )
            .mark_circle(
                size=140,
                filled=True
            )
        )


        # Босоо шулуун 
        vline = (
            alt.Chart(chart_df)
            .mark_rule(color="#aaaaaa", strokeWidth=1.2)  
            .encode(
                x='time_dt:T',
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            )
            .transform_filter(hover)
        )

        
        # ===== 🔟 ҮНДСЭН ГРАФИК =====
        if is_credit_supply and all([household_bar, corporate_bar, household_line, corporate_line]):
            # 🔥 CREDIT SUPPLY SPECIALIZED CHART
            
            # 1️⃣ STACKED BAR CHART
            bars = (
                alt.Chart(chart_df)
                .transform_fold(
                    [household_bar, corporate_bar],
                    as_=["Indicator", "Value"]
                )
                .mark_bar(
                    filled=False,
                    stroke="#000000",
                    strokeWidth=2
                )
                .encode(
                    x=alt.X(
                        "time_dt:T",
                        title=None,
                        axis=x_axis,
                        scale=alt.Scale(zero=False, domain=mini_brush)
                    ),
                    y=alt.Y(
                        "Value:Q",
                        title=None,
                        stack="zero",
                        axis=alt.Axis(
                            grid=True,
                            gridOpacity=0.25,
                            domain=True,
                            labelFontSize=11,
                            offset=5,
                            orient="left"
                        )
                    ),
                    stroke=alt.Stroke(
                        "Indicator:N",
                        scale=alt.Scale(
                            domain=[household_bar, corporate_bar],
                            range=["#fbbf24", "#3b82f6"]
                        ),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("time_dt:T", title="Time", format="%Y-Q%q"),
                        alt.Tooltip("Indicator:N"),
                        alt.Tooltip("Value:Q", format=",.2f", title="Value")
                    ]
                )
            )
            
            # 2️⃣ HOUSEHOLD LINE (yellow)
            line_household = (
                alt.Chart(chart_df)
                .mark_line(
                    strokeWidth=2.5,
                    color="#fbbf24",
                    interpolate="monotone"
                )
                .encode(
                    x=alt.X("time_dt:T", title=None, axis=None),
                    y=alt.Y(
                        f"{household_line}:Q",
                        title=None,
                        axis=alt.Axis(
                            orient="right",
                            grid=False,
                            labelColor="#fbbf24",
                            labelFontSize=11
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("time_dt:T", title="Time", format="%Y-Q%q"),
                        alt.Tooltip(f"{household_line}:Q", format=",.2f")
                    ]
                )
            )
            
            # 3️⃣ CORPORATE LINE (blue dashed)
            line_corporate = (
                alt.Chart(chart_df)
                .mark_line(
                    strokeWidth=2.5,
                    strokeDash=[5, 5],
                    color="#3b82f6",
                    interpolate="monotone"
                )
                .encode(
                    x=alt.X("time_dt:T", title=None, axis=None),
                    y=alt.Y(
                        f"{corporate_line}:Q",
                        title=None,
                        axis=None
                    ),
                    tooltip=[
                        alt.Tooltip("time_dt:T", title="Time", format="%Y-Q%q"),
                        alt.Tooltip(f"{corporate_line}:Q", format=",.2f")
                    ]
                )
            )
            
            # 4️⃣ COMBINE ALL LAYERS
            main_chart = (
                alt.layer(bars, line_household, line_corporate)
                .resolve_scale(y='independent')
                .properties(
                    height=400,
                    width=850
                )
                .add_params(zoom_brush)
            )
            
        else:
            # 🔍 STANDARD LINE CHART (for all other groups)
            main_chart = (
                alt.layer(
                    line,
                    vline,
                    points,
                    last_point
                )
                .properties(
                    height=400,
                    width=850
                )
                .add_params(zoom_brush)
            )
        # MINI CHART ИЙН ШУГАМ - ЯМАР Ч ZOOM, PAN ХИЙХГҮЙ
        mini_line = (
            alt.Chart(chart_df)
            .transform_fold(
                valid_indicators,
                as_=["Indicator", "Value"]
            )
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("time_dt:T", 
                        axis=None,
                        # 🔥 MINI CHART НЬ ХЭЗЭЭ Ч ZOOM ХИЙХГҮЙ - БҮХ ӨГӨГДӨЛ ҮРГЭЛЖ ХАРАГДДАГ
                        scale=alt.Scale(domain=[chart_df["time_dt"].min(), chart_df["time_dt"].max()])
                ),
                y=alt.Y(
                    "Value:Q",
                    axis=alt.Axis(
                        labels=False,
                        ticks=False,
                        grid=False,
                        domain=False
                    ),
                    scale=alt.Scale(zero=False)
                ),
                color=alt.Color("Indicator:N", legend=None)
            )
        )
        
        # MINI WINDOW - ЗӨВХӨН zoom_brush-ийн domain-ыг ХАРУУЛНА
        # zoom_brush өөрчлөгдөх бүрт window шинэчлэгдэнэ
        mini_window = (
            alt.Chart(chart_df)
            .mark_rect(
                fill="#888888",         
                fillOpacity=0.15,
                stroke="#777777",
                strokeWidth=1.2
            )
            .encode(
                x=alt.X('min(time_dt):T', title=None),
                x2=alt.X2('max(time_dt):T')
            )
            .transform_filter(zoom_brush)  # 🔥 zoom_brush-ын domain-ыг ашиглана
        )
        
        mini_chart = (
            alt.layer(
                mini_line,
                mini_window
            )
            .properties(
                height=60,
                width=800
            )
            # ✅ MINI CHART ДЭЭР PAN ХИЙХ БОЛОМЖТОЙ (WINDOW-Г ЧИРЖ БАЙРЛУУЛАХ)
            .add_params(mini_brush)
        )

        if is_credit_supply and all([household_bar, corporate_bar, household_line, corporate_line]):
            # Custom legend for Credit Supply
            all_inds = [household_bar, corporate_bar, household_line, corporate_line]
            legend_chart = (
                alt.Chart(
                    pd.DataFrame({
                        "Indicator": all_inds,
                        "Order": [1, 2, 3, 4]
                    })
                )
                .mark_point(size=0, opacity=0)
                .encode(
                    color=alt.Color(
                        "Indicator:N",
                        scale=alt.Scale(
                            domain=all_inds,
                            range=["#fbbf24", "#3b82f6", "#fbbf24", "#3b82f6"]
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            direction="horizontal",
                            title=None,
                            labelLimit=200,
                            labelFontSize=10,
                            symbolSize=80,
                            symbolType="square",
                            columnPadding=8,
                            padding=0,
                            offset=2
                        )
                    )
                )
            )
            
            main_chart = alt.layer(main_chart, legend_chart)
        # ===== 1️⃣2️⃣ НЭГТГЭСЭН ГРАФИК =====
        final_chart = (
            alt.vconcat(
                main_chart,
                mini_chart,
                spacing=20  # ✅ ЯГ ӨМНӨХ ШИГЭЭ 20
            )
            .resolve_scale(
                x='independent',
                color='shared'
            )
            .properties(
                # ✅ ЯГ ӨМНӨХ ШИГЭЭ PADDING
                padding={"left": 0, "top": 20, "right": 20, "bottom": 20}
            )
            .configure_view(
                strokeWidth=0
            )
            .configure_axis(
                grid=True,
                gridColor='#e0e0e0',
                gridOpacity=0.3
            )
        )

       # ===== HEADER ROW: Chart title + download button =====
        header_col1, header_col2 = st.columns([6, 1])
        
        with header_col1:
            st.subheader("📈 Main chart")
        
        with header_col2:
            csv = chart_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV",
                data=csv,
                file_name="main_chart_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ===== MAIN CHART DISPLAY =====
        st.altair_chart(final_chart, use_container_width=True)


    
    def compute_group_kpis(df, indicators):
        stats = []
    
        for ind in indicators:
            if ind not in df.columns:
                continue
    
            series = df[["time", ind]].copy()
            series[ind] = pd.to_numeric(series[ind], errors="coerce")
    
            last_valid_idx = series[ind].last_valid_index()
            if last_valid_idx is None:
                continue
    
            raw_val = series.loc[last_valid_idx, ind]
    
            try:
                last_value = float(raw_val)
            except:
                continue
    
            last_date = str(series.loc[last_valid_idx, "time"])
    
            stats.append({
                "Indicator": ind,
                "Min": series[ind].min(),
                "Max": series[ind].max(),
                "Mean": series[ind].mean(),
                "Median": series[ind].median(),
                "Std": series[ind].std(),
                "Last": last_value,
                "Last date": last_date
            })
    
        return pd.DataFrame(stats)


    # ======================
    # 📊 KPI CALCULATION (INDICATOR LEVEL)
    # ======================
    
    group_indicators = [
        col[1] for col in df_data.columns
        if col[0] == group
    ]
    
    # 🔥 KPI ТООЦООЛОЛ: PERCENTAGE INDICATORS-Г 100-ААР ҮРЖҮҮЛСЭН DF
    kpi_chart_df = chart_df.copy()
    for ind in group_indicators:
        if ind in kpi_chart_df.columns and is_percentage_indicator(ind):
            kpi_chart_df[ind] = kpi_chart_df[ind] * 100
    
    # 🔹 БҮХ indicator-уудын KPI-г НЭГ УДАА бодно
    kpi_df = compute_group_kpis(kpi_chart_df, group_indicators)
    
    # 🔹 KPI-д харуулах PRIMARY indicator
    primary_indicator = selected[0]
    
    # 🔹 KPI-г салгах
    kpi_main = kpi_df[kpi_df["Indicator"] == primary_indicator]
    kpi_rest = kpi_df[kpi_df["Indicator"] != primary_indicator]
    
    # 🔥 FIXED STYLING (no purple gradient, compact)
    st.markdown("""
    <style>
    /* ===== KPI CARDS ===== */
    .kpi-card {
        background: linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.95),
            rgba(30, 41, 59, 0.85)
        );
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 12px;
        padding: 16px 18px;
        margin: 8px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(59,130,246,0.25);
        border-color: rgba(59,130,246,0.6);
    }
    
    .kpi-label {
        font-size: 10px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-family: 'Monaco', 'Courier New', monospace;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #60a5fa;
        font-family: 'Monaco', 'Courier New', monospace;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 8px rgba(96,165,250,0.3);
        line-height: 1.2;
    }
    
    .kpi-sub {
        font-size: 11px;
        color: #cbd5e1;
        opacity: 0.7;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* ===== HEADER STYLING ===== */
    .kpi-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 20px 0 16px 0;
        padding: 12px 16px;
        background: linear-gradient(
            90deg,
            rgba(59,130,246,0.1),
            rgba(139,92,246,0.05)
        );
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }
    
    .kpi-header-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    .kpi-header-indicator {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60a5fa;
        font-family: 'Monaco', 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    def format_kpi(indicator, value):
        if value is None or pd.isna(value):
            return "N/A"
        
        if is_percentage_indicator(indicator):
            return f"{value:.2f}%"  # ✅ АЛЬ ХЭДИЙН 100-ААР ҮРЖИГДСЭН
        else:
            return f"{value:,.2f}"


    # ===== KPI CARD HELPER
    def kpi_card(label, value, sublabel=None):
        sub = ""
        if sublabel is not None:
            sub = f"<div class='kpi-sub'>{str(sublabel)}</div>"
        
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {sub}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 🔥 HEADER
    st.markdown(
        f"""
        <div class="kpi-header">
            <span class="kpi-header-title">📌 Indicator-level KPIs</span>
            <span style="opacity: 0.4;">→</span>
            <span class="kpi-header-indicator">{primary_indicator}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if kpi_main.empty:
        st.info("No KPI data available.")
        st.stop()
    
    row = kpi_main.iloc[0]
    
    # 🔽 KPI CARDS
    cols = st.columns(6)
    
    with cols[0]:
        last_date = str(row["Last date"]).split('\n')[0].split('Name:')[0].strip()
        kpi_card(
            "LAST VALUE",
            format_kpi(primary_indicator, row["Last"]),
            last_date
        )
    
    with cols[1]:
        kpi_card("MEAN", format_kpi(primary_indicator, row["Mean"]))
    
    with cols[2]:
        kpi_card("MEDIAN", format_kpi(primary_indicator, row["Median"]))
    
    with cols[3]:
        kpi_card("MINIMUM VALUE", format_kpi(primary_indicator, row["Min"]))
    
    with cols[4]:
        kpi_card("MAXIMUM VALUE", format_kpi(primary_indicator, row["Max"]))
    
    with cols[5]:
        kpi_card("STD (VOLATILITY)", format_kpi(primary_indicator, row["Std"]))

    
    # ======================
    # 📋 OPTIONAL — Indicator-level KPI TABLE
    # ======================
    if not kpi_rest.empty:
        with st.expander("📋 Indicator-level statistics"):
            st.dataframe(
                kpi_rest
                .set_index("Indicator")
                .round(2),
                use_container_width=True
            )
    
    # ======================
    # 📉 CHANGE SUMMARY — ENHANCED PRO STYLE
    # ======================
    st.markdown("### 📉 Change summary")
    
    # 🔥 Change summary-д ашиглах indicator-ууд
    if selected:
        change_indicators = selected
    else:
        change_indicators = [
            col[1] for col in df_data.columns
            if col[0] == group and not pd.isna(col[1])
        ]
    
    if not group_indicators:
        st.caption("No indicators in this group.")
    else:
        cards_html = ""
        
        for ind in change_indicators:
            tmp = pd.DataFrame({
                "x": series["time"],
                ind: df_data[(group, ind)].values
            })
            
            if not tmp[ind].isna().all():
                changes = compute_changes(tmp, ind, freq)
            else:
                changes = None
            
            if changes:
                # 🔹 Өнгөний логик (up=green, down=red)
                def render_metric(label, value):
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        return f"<span class='metric-item metric-neutral'><span class='metric-label'>{label}</span><span class='metric-value'>N/A</span></span>"
                    
                    cls = "metric-up" if value > 0 else "metric-down" if value < 0 else "metric-neutral"
                    arrow = "▲" if value > 0 else "▼" if value < 0 else "─"
                    
                    return (
                        f"<span class='metric-item {cls}'>"
                        f"<span class='metric-label'>{label}</span>"
                        f"<span class='metric-value'>{arrow} {value:.1f}%</span>"
                        f"</span>"
                    )
                
                cards_html += f"""
                <div class="change-card-pro">
                    <div class="change-title-pro">{ind}</div>
                    <div class="change-metrics-pro">
                        {render_metric("YoY", changes.get("yoy"))}
                        {render_metric("YTD", changes.get("ytd"))}
                        {render_metric("Prev", changes.get("prev"))}
                    </div>
                </div>
                """
        
        # ✅ ENHANCED STYLING
        if cards_html:
            components.html(
                """
                <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                .change-container-pro {
                    display: inline-flex;
                    flex-wrap: wrap;  /* Олон мөр болгох */
                    gap: 6px;
                    padding: 8px 0;
                    max-height: none;
                    overflow: visible;
                }
                .change-grid-pro {
                    display: inline-flex;
                    flex-wrap: wrap;
                    gap: 6px;
                    padding: 8px 4px;
                    overflow: visible;
                }
                .change-card-pro {
                    width: fit-content;          
                    min-width: unset;          
                    max-width: unset;            
                    flex: 0 0 auto;             
                    padding: 16px 18px;
                    background: linear-gradient(
                        135deg,
                        rgba(19, 47, 94, 0.85),
                        rgba(15, 41, 83, 0.75)
                    );
                    border: 1px solid rgba(20, 52, 124, 0.35);
                    border-radius: 10px;
                    transition: all 0.25s ease;
                }
                
                .change-card-pro:hover {
                    transform: translateY(-3px);
                    border-color: rgba(20, 52, 124, 0.6);
                    box-shadow: 0 6px 20px rgba(20, 52, 124, 0.25);
                }
                
                .change-title-pro {
                    font-size: 14px;
                    font-weight: 700;
                    color: #e2e8f0;
                    margin-bottom: 14px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid rgba(148,163,184,0.2);
                }
                
                .change-metrics-pro {
                    display: inline-flex;
                    flex-direction: column;
                    gap: 4px;
                }
                
                .metric-item {
                    width: fit-content;
                    max-width: 100%;
                    display: inline-flex;
                    justify-content: flex-start;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 8px;
                    background: rgba(30,41,59,0.6);
                    border-radius: 8px;
                    border-left: 3px solid transparent;
                    transition: all 0.2s ease;
                }
                
                .metric-item:hover {
                    background: rgba(30,41,59,0.9);
                }
                
                .metric-label {
                    font-size: 11px;
                    font-weight: 600;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.01em;
                }
                
                .metric-value {
                    font-size: 15px;
                    font-weight: 700;
                    font-family: 'Monaco', 'Courier New', sans-serif;
                }
                
                .metric-up {
                    border-left-color: #22c55e;
                }
                
                .metric-up .metric-value {
                    color: #22c55e;
                    text-shadow: 0 0 8px rgba(34,197,94,0.4);
                }
                
                .metric-down {
                    border-left-color: #ef4444;
                }
                
                .metric-down .metric-value {
                    color: #ef4444;
                    text-shadow: 0 0 8px rgba(239,68,68,0.4);
                }
                
                .metric-neutral .metric-value {
                    color: #94a3b8;
                }
                
                /* Scrollbar */
                ::-webkit-scrollbar {
                    height: 8px;
                }
                
                ::-webkit-scrollbar-track {
                    background: rgba(30,41,59,0.5);
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: rgba(148,163,184,0.4);
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: rgba(148,163,184,0.6);
                }
                </style>
                
                <div class="change-grid-pro">
                """ + cards_html + """
                </div>
                """,
                height=200
            )
        else:
            st.caption("No data yet")
# ======================
# SMALL MULTIPLE CHART
# ======================
def small_multiple_chart(df, indicator):
    import altair as alt

    return (
        alt.Chart(df)
        .transform_filter(
            alt.datum[indicator] != None
        )
        .mark_line(
            strokeWidth=2,
            interpolate="linear"
        )
        .encode(
            x=alt.X(
                "x:N",
                title=None,
                axis=alt.Axis(
                    labels=False,
                    ticks=False,
                    grid=False
                )
            ),
            y=alt.Y(
                f"{indicator}:Q",
                title=None,
                axis=alt.Axis(
                    grid=True,
                    gridOpacity=0.2,
                    domain=False
                )
            ),
            tooltip=[
                alt.Tooltip("x:N"),
                alt.Tooltip(f"{indicator}:Q", format=",.1f")
            ]
        )
        .properties(
            height=320,
            title=alt.TitleParams(
                text=indicator,
                anchor="start",
                fontSize=14,
                offset=6
            ),
            background="transparent"
        )
    )
# ======================
# 📊 ALL INDICATOR GROUPS — SMALL MULTIPLES (FULL WIDTH)
# ======================

st.markdown("### 📊 All indicator groups")

import altair as alt

# бүх group-ууд
all_groups = df_data.columns.get_level_values(0).unique()

NUM_COLS = 4
rows = [
    all_groups[i:i + NUM_COLS]
    for i in range(0, len(all_groups), NUM_COLS)
]

def group_chart(group_name):
    import altair as alt

    # 1️⃣ тухайн group-ийн бүх indicator
    inds = [
        col[1] for col in df_data.columns
        if col[0] == group_name and not pd.isna(col[1])
    ]

    # 2️⃣ суурь dataframe (YEAR + INDICATORS)
    gdf = pd.DataFrame({
        "time": series["time"].values
    })
    
    # 🔥 indicator-уудыг НЭМНЭ
    for ind in inds:
        if (group_name, ind) in df_data.columns:
            gdf[ind] = df_data[(group_name, ind)].values
    # ⛔ SMALL CHART — 2020 оноос хойш
    gdf = gdf[gdf["time"] >= "2020"]



    # ✅ 5️⃣ өгөгдөлтэй indicator-ууд
    valid_inds = [
        c for c in inds
        if c in gdf.columns and not gdf[c].isna().all()
    ]

    # 6️⃣ BASE CHART (үргэлж харагдана)
    base = alt.Chart(gdf).encode(
        x=alt.X(
            "time:N",
            title=None,
            sort="ascending",
            axis=alt.Axis(
                labelAngle=0,
                grid=False,
                labelFontSize=11,
                labelExpr="substring(datum.value, 0, 4)"
            )
        )
    ).properties(
        height=320,
        padding={"top": 6, "bottom": 0, "left": 6, "right": 6},
        title=alt.TitleParams(
            text=group_name,
            anchor="start",
            fontSize=14,
            offset=6
        ),
        background="transparent"
    )


    # 7️⃣ ХЭРВЭЭ ӨГӨГДӨЛ БАЙХГҮЙ БОЛ
    if not valid_inds:
        return (
            alt.Chart(
                pd.DataFrame({"x": [0], "y": [0], "label": ["No data yet"]})
            )
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=13,
                color="#94a3b8"
            )
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                text="label:N"
            )
            .properties(
                height=320,
                title=alt.TitleParams(
                    text=group_name,
                    anchor="start",
                    fontSize=14,
                    offset=6
                ),
                background="transparent"
            )
        )
    
    # 8️⃣ ХЭРВЭЭ ӨГӨГДӨЛ БАЙВАЛ LINE
    lines = base.transform_fold(
        valid_inds,
        as_=["Indicator", "Value"]
    ).mark_line(strokeWidth=2).encode(
        y=alt.Y(
            "Value:Q",
            title=None,
            axis=alt.Axis(
                grid=True,
                gridColor="#334155",   
                gridOpacity=0.45,      
                gridWidth=1,           
                domain=False,
                tickColor="#475569",   # (сонголт)
                labelColor="#cbd5e1",  # (сонголт)
                titleColor="#e5e7eb",
                labelFontSize=11,
                titleFontSize=12
            )
        ),
        color=alt.Color(
            "Indicator:N", 
            legend=alt.Legend(
                orient="bottom",
                direction="horizontal",
                title=None,
                labelLimit=150,
                labelFontSize=11,
                symbolSize=80,
                symbolStrokeWidth=2,
                columnPadding=4,
                padding=0,
                offset=2
            )
        ),
        tooltip=[
            alt.Tooltip("time:N"),
            alt.Tooltip("Indicator:N"),
            alt.Tooltip("Value:Q", format=",.2f")
        ]
    )
    

    # ======================
    # 🔥 CREDIT SUPPLY CHART (QUARTERLY ONLY)  
    # ======================
    if group_name == "Credit supply" and freq == "Quarterly":
        # 🔍 DEBUG: Indicator нэрүүдийг хэвлэх
        print(f"Available indicators: {valid_inds}")
        
        # Бүх indicator-ийн нэрийг case-insensitive шалгах
        household_bar = next((ind for ind in valid_inds if "issued" in ind.lower() and "household" in ind.lower()), None)
        corporate_bar = next((ind for ind in valid_inds if "issued" in ind.lower() and "corporate" in ind.lower()), None)
        household_line = next((ind for ind in valid_inds if "household" in ind.lower() and "supply" in ind.lower() and "issued" not in ind.lower()), None)
        corporate_line = next((ind for ind in valid_inds if "corporate" in ind.lower() and "supply" in ind.lower() and "issued" not in ind.lower()), None)
        
        # 🔍 DEBUG: Олдсон нэрүүдийг хэвлэх
        print(f"Household bar: {household_bar}")
        print(f"Corporate bar: {corporate_bar}")
        print(f"Household line: {household_line}")
        print(f"Corporate line: {corporate_line}")
        
        if all([household_bar, corporate_bar, household_line, corporate_line]):
            # 1️⃣ STACKED BAR CHART
            bars = (
                alt.Chart(gdf)
                .transform_fold(
                    [household_bar, corporate_bar],
                    as_=["Indicator", "Value"]
                )
                .mark_bar(
                    filled=False,         
                    stroke="#000000",       
                    strokeWidth=2           
                )
                .encode(
                    x=alt.X(
                        "time:N",
                        title=None,
                        sort="ascending",
                        axis=alt.Axis(
                            labelAngle=0,
                            grid=False,
                            labelFontSize=10,
                            labelExpr="split(datum.value, '-')[1] + '\\n' + split(datum.value, '-')[0]",
                            labelPadding=8,
                            domain=True
                        )
                    ),
                    y=alt.Y(
                        "Value:Q",
                        title=None,
                        stack="zero",
                        axis=alt.Axis(
                            grid=True,
                            gridColor="#334155",
                            gridOpacity=0.45,
                            labelColor="#cbd5e1",
                            labelFontSize=11,
                            orient="left"
                        )
                    ),
                    stroke=alt.Stroke(       # ✅ Өнгийг хүрээнд ашиглана
                        "Indicator:N",
                        scale=alt.Scale(
                            domain=[household_bar, corporate_bar],
                            range=["#fbbf24", "#3b82f6"]
                        ),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("time:N"),
                        alt.Tooltip("Indicator:N"),
                        alt.Tooltip("Value:Q", format=",.2f")
                    ]
                )
            )
            
            # 2️⃣ HOUSEHOLD LINE (шар)
            line_household = (
                alt.Chart(gdf)
                .mark_line(
                    strokeWidth=2.5,
                    color="#fbbf24",
                    interpolate="monotone"
                    #point=alt.OverlayMarkDef(size=60, filled=True, color="#fbbf24")
                )
                .encode(
                    x=alt.X("time:N", title=None, sort="ascending", axis=None),
                    y=alt.Y(
                        f"{household_line}:Q",
                        title=None,
                        axis=alt.Axis(
                            orient="right",
                            grid=False,
                            labelColor="#fbbf24",
                            labelFontSize=11
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("time:N"),
                        alt.Tooltip(f"{household_line}:Q", format=",.2f")
                    ]
                )
            )
            
            # 3️⃣ CORPORATE LINE (цэнхэр тасархай)
            line_corporate = (
                alt.Chart(gdf)
                .mark_line(
                    strokeWidth=2.5,
                    strokeDash=[5, 5],
                    color="#3b82f6",
                    interpolate="monotone"
                    #point=alt.OverlayMarkDef(size=60, filled=True, color="#3b82f6")
                )
                .encode(
                    x=alt.X("time:N", title=None, sort="ascending", axis=None),
                    y=alt.Y(
                        f"{corporate_line}:Q",
                        title=None,
                        axis=None
                    ),
                    tooltip=[
                        alt.Tooltip("time:N"),
                        alt.Tooltip(f"{corporate_line}:Q", format=",.2f")
                    ]
                )
            )
            
            # 4️⃣ COMBINE
            combined = (
                alt.layer(bars, line_household, line_corporate)
                .resolve_scale(y='independent')
            )
            
            # 5️⃣ LEGEND
            all_inds = [household_bar, corporate_bar, household_line, corporate_line]
            legend_chart = (
                alt.Chart(
                    pd.DataFrame({
                        "Indicator": all_inds,
                        "Order": [1, 2, 3, 4]
                    })
                )
                .mark_point(size=0, opacity=0)
                .encode(
                    color=alt.Color(
                        "Indicator:N",
                        scale=alt.Scale(
                            domain=all_inds,
                            range=["#fbbf24", "#3b82f6", "#fbbf24", "#3b82f6"]
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            direction="horizontal",
                            title=None,
                            labelLimit=200,
                            labelFontSize=10,
                            symbolSize=80,
                            symbolType="square",
                            columnPadding=8,
                            padding=0,
                            offset=2
                        )
                    )
                )
            )
            
            # 6️⃣ FINAL
            final = (
                alt.layer(combined, legend_chart)
                .properties(
                    height=320,
                    width=800,
                    padding={"top": 6, "bottom": 0, "left": 6, "right": 6},
                    title=alt.TitleParams(
                        text=group_name,
                        anchor="start",
                        fontSize=14,
                        offset=6
                    ),
                    background="transparent"
                )
            )
            
            return final
    
    return lines


for row in rows:
    cols = st.columns(NUM_COLS, gap="small")
    for col, grp in zip(cols, row):
        with col:
            with st.container(border=True):
                chart = group_chart(grp)
                if chart is not None:
                    st.altair_chart(chart, use_container_width=True)
# ======================
# 📄 RAW DATA — INDICATOR GROUP LEVEL
# ======================
with st.expander(f"📄 Raw data — {group} group"):
    
    # 1️⃣ тухайн group-д хамаарах бүх indicator
    group_cols = [
        col[1] for col in df_data.columns
        if col[0] == group and not pd.isna(col[1])
    ]

    if not group_cols:
        st.info("No indicators in this group.")
    else:
        raw_group_df = pd.DataFrame({
            "time": series["time"]
        })

        # 2️⃣ indicator-уудыг нэмэх
        for ind in group_cols:
            if (group, ind) in df_data.columns:
                raw_group_df[ind] = df_data[(group, ind)].values

        # 3️⃣ цэгцлэх
        raw_group_df = (
            raw_group_df
            .dropna(how="all", subset=group_cols)
            .sort_values("time")
            .reset_index(drop=True)
        )

        st.dataframe(
            raw_group_df,
            use_container_width=True
        )
