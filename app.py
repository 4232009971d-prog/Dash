import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from dataclasses import dataclass, field
from typing import Any

st.set_page_config(
    page_title="InsightFlow",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0F1117; }
section[data-testid="stSidebar"] {
  background: #1A1D2E;
  border-right: 1px solid #2D3150;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.kpi-card {
  background: #1E2235;
  border: 1px solid #2D3150;
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #6366F1, #8B5CF6);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
.kpi-value {
  font-size: 1.7rem;
  font-weight: 800;
  color: #F1F5F9;
  letter-spacing: -1px;
  line-height: 1;
}
.kpi-label {
  font-size: 0.75rem;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 0.35rem;
}
.insight-card {
  background: #1E2235;
  border: 1px solid #2D3150;
  border-left: 3px solid #6366F1;
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.75rem;
  display: flex;
  gap: 0.75rem;
}
.insight-card.high { border-left-color: #EF4444; }
.insight-card.medium { border-left-color: #F59E0B; }
.insight-card.low { border-left-color: #10B981; }
.insight-text { font-size: 0.88rem; color: #94A3B8; line-height: 1.5; }
.section-heading {
  font-size: 0.75rem;
  font-weight: 700;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin: 2rem 0 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-heading::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #2D3150;
}
.stButton > button {
  background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Synonym Dictionary ────────────────────────────────────────────────────────
SYNONYMS = {
    "revenue": ["revenue","sales","amount","net amount","invoice amount",
                "total amount","total sales","sale amount","turnover",
                "income","value","total value","line total"],
    "quantity": ["quantity","qty","units","pieces","pcs","count","volume"],
    "date":     ["date","order date","invoice date","sale date","transaction date",
                 "ship date","delivery date","dt","period"],
    "customer": ["customer name","client name","customer","client","buyer","account name"],
    "product":  ["product name","product","item","item name","description","article"],
    "region":   ["region","territory","zone","area","district","market"],
    "country":  ["country","nation","country name"],
    "profit":   ["profit","net profit","gross profit","earnings","margin amount"],
    "cost":     ["cost","cost price","cogs","purchase price"],
}

def detect_col(columns, role):
    for col in columns:
        clean = col.lower().strip().replace("_"," ")
        if clean in SYNONYMS.get(role, []):
            return col
    for col in columns:
        clean = col.lower().strip().replace("_"," ")
        for alias in SYNONYMS.get(role, []):
            if alias in clean or clean in alias:
                return col
    return None

# ── Data Cleaning ─────────────────────────────────────────────────────────────
def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    blank_rows = int(df.isnull().all(axis=1).sum())
    df.dropna(how="all", inplace=True)
    dup_count = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            df[col].replace(["","N/A","NA","None","NULL","-"], np.nan, inplace=True)
    null_pct = df.isnull().mean() * 100
    score = max(0, 100 - null_pct.mean() - (dup_count / max(len(df),1) * 10))
    return df, dup_count, blank_rows, round(score, 1)

# ── KPI Computation ───────────────────────────────────────────────────────────
def compute_kpis(df, cols):
    kpis = []
    rev_col = cols.get("revenue")
    qty_col = cols.get("quantity")
    cust_col = cols.get("customer")
    prod_col = cols.get("product")
    profit_col = cols.get("profit")
    date_col = cols.get("date")

    if rev_col:
        rev = pd.to_numeric(df[rev_col], errors="coerce")
        total = rev.sum()
        kpis.append(("💰","Total Revenue", fmt_currency(total)))
        if len(df) > 0:
            kpis.append(("🛒","Avg Order Value", fmt_currency(total/len(df))))

    kpis.append(("📋","Total Orders", f"{len(df):,}"))

    if cust_col:
        kpis.append(("👥","Unique Customers", f"{df[cust_col].nunique():,}"))

    if prod_col:
        kpis.append(("🏷️","Unique Products", f"{df[prod_col].nunique():,}"))

    if profit_col and rev_col:
        profit = pd.to_numeric(df[profit_col], errors="coerce").sum()
        rev_total = pd.to_numeric(df[rev_col], errors="coerce").sum()
        if rev_total > 0:
            margin = profit / rev_total * 100
            kpis.append(("💹","Profit Margin", f"{margin:.1f}%"))

    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(dates) > 0:
                kpis.append(("📅","Date Range",
                    f"{dates.min().date()} – {dates.max().date()}"))
        except:
            pass

    return kpis

def fmt_currency(v):
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000: return f"${v/1_000:.1f}K"
    return f"${v:,.2f}"

# ── Chart Generation ──────────────────────────────────────────────────────────
PALETTE = ["#6366F1","#8B5CF6","#EC4899","#14B8A6","#F59E0B","#10B981","#3B82F6"]
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", size=11),
    margin=dict(l=20,r=20,t=30,b=20),
)

def apply_theme(fig):
    fig.update_layout(**THEME)
    fig.update_xaxes(gridcolor="#1E293B", linecolor="#334155")
    fig.update_yaxes(gridcolor="#1E293B", linecolor="#334155")
    return fig

def generate_charts(df, cols):
    charts = []
    rev_col  = cols.get("revenue")
    date_col = cols.get("date")
    prod_col = cols.get("product")
    cust_col = cols.get("customer")
    reg_col  = cols.get("region")
    qty_col  = cols.get("quantity")

    # Revenue over time
    if date_col and rev_col:
        try:
            df_t = df[[date_col, rev_col]].copy()
            df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
            df_t[rev_col]  = pd.to_numeric(df_t[rev_col], errors="coerce")
            df_t.dropna(inplace=True)
            df_t["Month"] = df_t[date_col].dt.to_period("M").dt.to_timestamp()
            monthly = df_t.groupby("Month")[rev_col].sum().reset_index()
            if len(monthly) >= 2:
                fig = px.line(monthly, x="Month", y=rev_col,
                    markers=True, color_discrete_sequence=[PALETTE[0]])
                fig.update_traces(fill="tozeroy",
                    fillcolor="rgba(99,102,241,0.1)", line=dict(width=2.5))
                charts.append(("Monthly Revenue Trend", apply_theme(fig)))
        except: pass

    # Top products
    if prod_col and rev_col:
        try:
            agg = df.groupby(prod_col)[rev_col].sum().sort_values(ascending=False).head(10).reset_index()
            agg[rev_col] = pd.to_numeric(agg[rev_col], errors="coerce")
            fig = px.bar(agg, x=rev_col, y=prod_col, orientation="h",
                color_discrete_sequence=[PALETTE[1]])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            charts.append(("Top 10 Products by Revenue", apply_theme(fig)))
        except: pass

    # Top customers
    if cust_col and rev_col:
        try:
            agg = df.groupby(cust_col)[rev_col].sum().sort_values(ascending=False).head(10).reset_index()
            agg[rev_col] = pd.to_numeric(agg[rev_col], errors="coerce")
            fig = px.bar(agg, x=rev_col, y=cust_col, orientation="h",
                color_discrete_sequence=[PALETTE[2]])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            charts.append(("Top 10 Customers by Revenue", apply_theme(fig)))
        except: pass

    # Region donut
    if reg_col and rev_col:
        try:
            agg = df.groupby(reg_col)[rev_col].sum().reset_index()
            agg[rev_col] = pd.to_numeric(agg[rev_col], errors="coerce")
            fig = px.pie(agg, names=reg_col, values=rev_col,
                color_discrete_sequence=PALETTE, hole=0.4)
            charts.append(("Revenue by Region", apply_theme(fig)))
        except: pass

    # Units sold
    if prod_col and qty_col:
        try:
            agg = df.groupby(prod_col)[qty_col].sum().sort_values(ascending=False).head(10).reset_index()
            agg[qty_col] = pd.to_numeric(agg[qty_col], errors="coerce")
            fig = px.bar(agg, x=prod_col, y=qty_col,
                color_discrete_sequence=[PALETTE[4]])
            charts.append(("Top Products by Units Sold", apply_theme(fig)))
        except: pass

    return charts

# ── Insights ──────────────────────────────────────────────────────────────────
def generate_insights(df, cols, dup_count, quality_score):
    insights = []
    rev_col  = cols.get("revenue")
    date_col = cols.get("date")
    prod_col = cols.get("product")
    cust_col = cols.get("customer")

    if rev_col:
        rev = pd.to_numeric(df[rev_col], errors="coerce")
        total = rev.sum()

        if date_col:
            try:
                df_t = df[[date_col, rev_col]].copy()
                df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
                df_t[rev_col]  = pd.to_numeric(df_t[rev_col], errors="coerce")
                df_t.dropna(inplace=True)
                df_t["Month"] = df_t[date_col].dt.to_period("M")
                monthly = df_t.groupby("Month")[rev_col].sum()
                if len(monthly) >= 2:
                    last = monthly.iloc[-1]
                    prev = monthly.iloc[-2]
                    if prev > 0:
                        pct = (last - prev) / prev * 100
                        if pct > 0:
                            insights.append(("high","📈",
                                f"Revenue grew {pct:.1f}% in the most recent month."))
                        else:
                            insights.append(("high","📉",
                                f"Revenue declined {abs(pct):.1f}% in the most recent month."))
            except: pass

        if prod_col:
            try:
                agg = pd.to_numeric(
                    df.groupby(prod_col)[rev_col].sum(), errors="coerce"
                ).sort_values(ascending=False)
                if len(agg) > 0 and agg.sum() > 0:
                    top_pct = agg.iloc[0] / agg.sum() * 100
                    if top_pct > 30:
                        insights.append(("high","⚠️",
                            f"'{agg.index[0]}' contributes {top_pct:.0f}% of total revenue — concentration risk."))
                    top3 = agg.head(3).sum() / agg.sum() * 100
                    if top3 > 60:
                        insights.append(("medium","🏆",
                            f"Top 3 products account for {top3:.0f}% of total revenue."))
                    low = int((agg / agg.sum() * 100 < 1).sum())
                    if low > 0:
                        insights.append(("low","🔍",
                            f"{low} products contribute less than 1% of revenue each."))
            except: pass

        if cust_col:
            try:
                agg = pd.to_numeric(
                    df.groupby(cust_col)[rev_col].sum(), errors="coerce"
                ).sort_values(ascending=False)
                if len(agg) > 0 and agg.sum() > 0:
                    top_pct = agg.iloc[0] / agg.sum() * 100
                    if top_pct > 25:
                        insights.append(("high","⭐",
                            f"'{agg.index[0]}' is your top customer at {top_pct:.0f}% of revenue."))
            except: pass

    if dup_count > 0:
        insights.append(("medium","🔄",
            f"{dup_count} duplicate rows were removed during cleaning."))
    if quality_score < 70:
        insights.append(("high","⚠️",
            f"Data quality score is {quality_score}/100 — review missing values."))

    return insights

# ── Session State ─────────────────────────────────────────────────────────────
for k,v in {"page":"upload","df":None,"cols":{},"kpis":[],
            "charts":[],"insights":[],"dup":0,"blank":0,
            "score":100,"fname":"","chat":[]}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 16px;border-bottom:1px solid #2D3150;margin-bottom:12px'>
      <div style='font-size:22px;font-weight:800;background:linear-gradient(135deg,#6366F1,#8B5CF6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>⚡ InsightFlow</div>
      <div style='color:#475569;font-size:11px;letter-spacing:1px;text-transform:uppercase;
        margin-top:3px'>AI Dashboard Generator</div>
    </div>
    """, unsafe_allow_html=True)

    nav = [("upload","📤","Upload"),("dashboard","📊","Dashboard"),
           ("insights","💡","Insights"),("chat","💬","Chat with Data"),
           ("quality","🔬","Data Quality")]

    for pid, icon, label in nav:
        disabled = pid != "upload" and st.session_state.df is None
        if disabled:
            st.markdown(f'<div style="padding:10px 12px;color:#334155;font-size:14px">'
                       f'{icon} {label}</div>', unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid
                st.rerun()

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown(f"""
        <div style='padding:0 8px;font-size:12px;color:#475569'>
          <div>📁 {st.session_state.fname}</div>
          <div style='margin-top:4px'>{len(st.session_state.df):,} rows</div>
        </div>""", unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────
page = st.session_state.page

# UPLOAD
if page == "upload":
    st.markdown("## Upload Your Data")
    st.markdown('<p style="color:#475569">Excel or CSV · up to 50MB</p>',
                unsafe_allow_html=True)
    f = st.file_uploader("Drop file here", type=["xlsx","xls","csv"],
                         label_visibility="collapsed")
    if f:
        if st.button("⚡  Analyse Now"):
            with st.spinner("Analysing..."):
                try:
                    if f.name.endswith(".csv"):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f)

                    df, dup, blank, score = clean_data(df)

                    cols = {
                        "revenue":  detect_col(df.columns, "revenue"),
                        "quantity": detect_col(df.columns, "quantity"),
                        "date":     detect_col(df.columns, "date"),
                        "customer": detect_col(df.columns, "customer"),
                        "product":  detect_col(df.columns, "product"),
                        "region":   detect_col(df.columns, "region"),
                        "country":  detect_col(df.columns, "country"),
                        "profit":   detect_col(df.columns, "profit"),
                    }

                    kpis    = compute_kpis(df, cols)
                    charts  = generate_charts(df, cols)
                    insights = generate_insights(df, cols, dup, score)

                    st.session_state.df       = df
                    st.session_state.cols     = cols
                    st.session_state.kpis     = kpis
                    st.session_state.charts   = charts
                    st.session_state.insights = insights
                    st.session_state.dup      = dup
                    st.session_state.blank    = blank
                    st.session_state.score    = score
                    st.session_state.fname    = f.name
                    st.session_state.page     = "dashboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# DASHBOARD
elif page == "dashboard":
    df = st.session_state.df
    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown(f"## {st.session_state.fname}")
        st.markdown(f'<p style="color:#475569">{len(df):,} rows · '
                   f'{len(df.columns)} columns</p>', unsafe_allow_html=True)
    with col2:
        s = st.session_state.score
        c = "✅" if s>=80 else "⚠️" if s>=60 else "❌"
        color = "#10B981" if s>=80 else "#F59E0B" if s>=60 else "#EF4444"
        st.markdown(f'<div style="text-align:right;padding-top:20px">'
                   f'<span style="background:rgba(99,102,241,0.15);color:{color};'
                   f'padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600">'
                   f'{c} {s}/100 Quality</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Key Metrics</div>',
                unsafe_allow_html=True)

    kpis = st.session_state.kpis
    if kpis:
        cards = "".join(f'''<div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>''' for icon,label,val in kpis)
        st.markdown(f'<div class="kpi-grid">{cards}</div>',
                    unsafe_allow_html=True)

    charts = st.session_state.charts
    if charts:
        st.markdown('<div class="section-heading">Charts</div>',
                    unsafe_allow_html=True)
        cols_ui = st.columns(2, gap="medium")
        for i, (title, fig) in enumerate(charts):
            with cols_ui[i % 2]:
                st.markdown(f'<div style="font-size:11px;font-weight:700;'
                           f'color:#475569;text-transform:uppercase;'
                           f'letter-spacing:0.8px;margin-bottom:8px">'
                           f'{title}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True,
                               config={"displayModeBar": False})

    with st.expander("🗂️ Column Detection"):
        detected = {k:v for k,v in st.session_state.cols.items() if v}
        if detected:
            rows = [{"Role": k, "Column Found": v} for k,v in detected.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

# INSIGHTS
elif page == "insights":
    st.markdown("## AI Business Insights")
    insights = st.session_state.insights
    if not insights:
        st.info("Upload a file first.")
    else:
        for level, icon, text in insights:
            st.markdown(f'''<div class="insight-card {level}">
                <div style="font-size:1.1rem">{icon}</div>
                <div class="insight-text">{text}</div>
            </div>''', unsafe_allow_html=True)

# CHAT
elif page == "chat":
    st.markdown("## Chat with Your Data")

    suggestions = ["Total revenue?","How many customers?",
                   "How many orders?","Data quality score?"]
    cols_s = st.columns(4)
    for i, s in enumerate(suggestions):
        with cols_s[i]:
            if st.button(s, key=f"s{i}"):
                st.session_state.chat.append({"role":"user","content":s})
                df = st.session_state.df
                kpis = st.session_state.kpis
                q = s.lower()
                ans = "I don't have enough data to answer that."
                kpi_map = {k[1].lower():k[2] for k in kpis}
                if "revenue" in q:
                    ans = f"Total revenue is **{kpi_map.get('total revenue','N/A')}**"
                elif "customer" in q:
                    ans = f"There are **{kpi_map.get('unique customers','N/A')}** unique customers"
                elif "order" in q:
                    ans = f"There are **{kpi_map.get('total orders','N/A')}** total orders"
                elif "quality" in q:
                    ans = f"Data quality score is **{st.session_state.score}/100**"
                st.session_state.chat.append({"role":"assistant","content":ans})
                st.rerun()

    st.markdown("---")
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div style="background:rgba(99,102,241,0.1);border:1px solid '
                       f'rgba(99,102,241,0.2);border-radius:10px;padding:10px 14px;'
                       f'margin:6px 0 6px 15%;font-size:14px;color:#C7D2FE">'
                       f'{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#1E2235;border:1px solid #2D3150;'
                       f'border-radius:10px;padding:10px 14px;margin:6px 15% 6px 0;'
                       f'font-size:14px;color:#94A3B8">{msg["content"]}</div>',
                       unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything about your data...")
    if user_input:
        st.session_state.chat.append({"role":"user","content":user_input})
        kpis = st.session_state.kpis
        kpi_map = {k[1].lower():k[2] for k in kpis}
        q = user_input.lower()
        if "revenue" in q:
            ans = f"Total revenue is **{kpi_map.get('total revenue','N/A')}**"
        elif "customer" in q:
            ans = f"There are **{kpi_map.get('unique customers','N/A')}** unique customers"
        elif "order" in q:
            ans = f"There are **{kpi_map.get('total orders','N/A')}** total orders"
        elif "quality" in q:
            ans = f"Data quality score is **{st.session_state.score}/100**"
        else:
            top = [t for _,_,t in st.session_state.insights[:3]]
            ans = "Key insights:\n\n" + "\n\n".join(f"• {t}" for t in top) if top else "Upload a dataset first."
        st.session_state.chat.append({"role":"assistant","content":ans})
        st.rerun()

# QUALITY
elif page == "quality":
    st.markdown("## Data Quality Report")
    df = st.session_state.df
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Quality Score", f"{st.session_state.score}/100")
    c2.metric("Total Rows", f"{len(df):,}")
    c3.metric("Duplicates Removed", st.session_state.dup)
    c4.metric("Blank Rows Removed", st.session_state.blank)

    st.markdown("### Column Analysis")
    rows = []
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        status = "✅ Good" if null_pct < 5 else "⚠️ Issues" if null_pct < 50 else "❌ Critical"
        rows.append({"Column":col, "Type":str(df[col].dtype),
                    "Missing %":f"{null_pct:.1f}%",
                    "Unique":df[col].nunique(), "Status":status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
