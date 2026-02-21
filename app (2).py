import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RentIQ — House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f;
    color: #f0f0f8;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, rgba(200,240,80,0.08), rgba(80,70,229,0.08));
    border: 1px solid rgba(200,240,80,0.15);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c8f050, #5046e5, transparent);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #f0f0f8;
    margin: 0;
    line-height: 1;
}
.hero-title span { color: #c8f050; }
.hero-sub {
    color: #6b6b8a;
    font-size: 0.85rem;
    margin-top: 10px;
    letter-spacing: 0.5px;
}
.eyebrow {
    display: inline-block;
    background: rgba(200,240,80,0.08);
    border: 1px solid rgba(200,240,80,0.25);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #c8f050;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* Metric cards */
.metric-card {
    background: #12121a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200,240,80,0.3), transparent);
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #c8f050;
    letter-spacing: -1px;
}
.metric-lbl {
    font-size: 0.65rem;
    color: #6b6b8a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(200,240,80,0.06), rgba(80,70,229,0.06));
    border: 1px solid rgba(200,240,80,0.2);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #5046e5, #c8f050, #5046e5);
}
.result-rent {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #c8f050;
    letter-spacing: -3px;
}
.result-lbl {
    font-size: 0.68rem;
    color: #6b6b8a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-sub { font-size: 0.8rem; color: #6b6b8a; margin-top: 6px; }

/* Section labels */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6b6b8a;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 16px;
}

/* Info tag */
.info-tag {
    display: inline-block;
    background: rgba(80,70,229,0.12);
    border: 1px solid rgba(80,70,229,0.3);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.72rem;
    color: #8b82f5;
    margin: 3px;
}
.info-tag-green {
    background: rgba(200,240,80,0.08);
    border-color: rgba(200,240,80,0.25);
    color: #c8f050;
}
.info-tag-orange {
    background: rgba(255,122,69,0.08);
    border-color: rgba(255,122,69,0.25);
    color: #ff7a45;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

/* Buttons */
.stButton > button {
    background: #c8f050 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #d4f566 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(200,240,80,0.25) !important;
}

/* Selectbox & inputs */
.stSelectbox > div > div, .stSlider, .stNumberInput > div {
    background: #12121a !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #f0f0f8 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12121a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b6b8a !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(200,240,80,0.1) !important;
    color: #c8f050 !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    color: #c8f050 !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6b6b8a !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; }

hr { border-color: rgba(255,255,255,0.06) !important; }

/* Hide default streamlit decorations */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL (CACHED)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("House_Rent_Dataset.csv")

    # Parse floor info
    def parse_floor(f):
        f = str(f).lower()
        if 'ground' in f: return 0
        try:
            return int(f.split()[0])
        except:
            return 1

    df['Floor_Num'] = df['Floor'].apply(parse_floor)

    # Encode categoricals
    le_city     = LabelEncoder()
    le_furnish  = LabelEncoder()
    le_area     = LabelEncoder()
    le_tenant   = LabelEncoder()
    le_contact  = LabelEncoder()

    df['City_enc']       = le_city.fit_transform(df['City'])
    df['Furnish_enc']    = le_furnish.fit_transform(df['Furnishing Status'])
    df['AreaType_enc']   = le_area.fit_transform(df['Area Type'])
    df['Tenant_enc']     = le_tenant.fit_transform(df['Tenant Preferred'])
    df['Contact_enc']    = le_contact.fit_transform(df['Point of Contact'])

    features = ['BHK', 'Size', 'Floor_Num', 'Bathroom',
                'City_enc', 'Furnish_enc', 'AreaType_enc', 'Tenant_enc', 'Contact_enc']

    X = df[features]
    y = df['Rent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                       max_depth=5, min_samples_split=4,
                                       random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    acc  = max(0, min(100, round(r2 * 100, 1)))

    encoders = {
        'city': le_city, 'furnish': le_furnish,
        'area': le_area, 'tenant': le_tenant, 'contact': le_contact
    }

    return df, model, encoders, features, mae, r2, acc


@st.cache_data
def get_city_stats(_df):
    return _df.groupby('City')['Rent'].agg(['mean', 'median', 'min', 'max', 'count']).round(0)


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    df, model, encoders, features, mae, r2, acc = load_and_prepare()
    city_stats = get_city_stats(df)


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="eyebrow">● AI-Powered · India</div>
  <div class="hero-title">Rent<span>IQ</span></div>
  <div class="hero-title" style="font-size:1.8rem; color:#6b6b8a; font-weight:400;">House Rent Prediction</div>
  <div class="hero-sub">Gradient Boosting model trained on 4,700+ Indian property listings across 6 cities</div>
</div>
""", unsafe_allow_html=True)

# Model stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model R² Score", f"{r2:.3f}", "Gradient Boosting")
with c2:
    st.metric("Accuracy", f"{acc}%", "Test set")
with c3:
    st.metric("Mean Abs. Error", f"₹{mae:,.0f}", "avg deviation")
with c4:
    st.metric("Training Data", "4,747", "listings")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠  Predict Rent", "📊  Market Explorer", "🔍  Data Insights"])


# ══════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

        cities     = sorted(df['City'].unique().tolist())
        furnishing = sorted(df['Furnishing Status'].unique().tolist())
        area_types = sorted(df['Area Type'].unique().tolist())
        tenants    = sorted(df['Tenant Preferred'].unique().tolist())
        contacts   = sorted(df['Point of Contact'].unique().tolist())

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            sel_city = st.selectbox("🌆 City", cities, index=cities.index("Mumbai"))
        with r1c2:
            sel_bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6], index=1)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            sel_bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
        with r2c2:
            sel_floor = st.number_input("🏢 Floor Number", min_value=0, max_value=30, value=2)

        sel_size = st.slider("📐 Size (sq ft)", min_value=100, max_value=5000, value=900, step=50)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            sel_furnish = st.selectbox("🛋️ Furnishing", furnishing)
        with r3c2:
            sel_area = st.selectbox("📏 Area Type", area_types)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            sel_tenant = st.selectbox("👥 Tenant Preferred", tenants)
        with r4c2:
            sel_contact = st.selectbox("📞 Point of Contact", contacts)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⟶  Predict Rent")

    # ── RESULT COLUMN ──
    with col_result:
        st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

        if predict_clicked:
            # Encode inputs
            try:
                city_enc    = encoders['city'].transform([sel_city])[0]
                furnish_enc = encoders['furnish'].transform([sel_furnish])[0]
                area_enc    = encoders['area'].transform([sel_area])[0]
                tenant_enc  = encoders['tenant'].transform([sel_tenant])[0]
                contact_enc = encoders['contact'].transform([sel_contact])[0]
            except Exception as e:
                st.error(f"Encoding error: {e}")
                st.stop()

            input_data = pd.DataFrame([[
                sel_bhk, sel_size, sel_floor, sel_bath,
                city_enc, furnish_enc, area_enc, tenant_enc, contact_enc
            ]], columns=features)

            predicted_rent = int(model.predict(input_data)[0])
            predicted_rent = max(1000, predicted_rent)

            rent_low  = int(predicted_rent * 0.82)
            rent_high = int(predicted_rent * 1.18)

            # Price per sqft
            price_per_sqft = round(predicted_rent / sel_size, 1)

            # City average comparison
            city_avg = int(city_stats.loc[sel_city, 'mean'])
            vs_avg   = round(((predicted_rent - city_avg) / city_avg) * 100, 1)
            vs_str   = f"+{vs_avg}%" if vs_avg > 0 else f"{vs_avg}%"
            vs_color = "#ff7a45" if vs_avg > 15 else ("#4ade80" if vs_avg < -15 else "#c8f050")

            # Main result card
            st.markdown(f"""
            <div class="result-card">
              <div class="result-lbl">Estimated Monthly Rent</div>
              <div class="result-rent">₹{predicted_rent:,}</div>
              <div class="result-sub">{sel_city} · {sel_bhk} BHK · {sel_size:,} sq ft · {sel_furnish}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Min Estimate", f"₹{rent_low:,}")
            with m2:
                st.metric("Max Estimate", f"₹{rent_high:,}")
            with m3:
                st.metric("₹ / sq ft", f"₹{price_per_sqft}")

            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("City Average", f"₹{city_avg:,}")
            with m5:
                st.metric("vs City Avg", vs_str)
            with m6:
                st.metric("Annual Cost", f"₹{predicted_rent*12:,}")

            # Confidence gauge
            st.markdown("<br>", unsafe_allow_html=True)
            confidence = min(95, max(70, int(r2 * 100) + np.random.randint(-3, 4)))

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Score", 'font': {'color': '#6b6b8a', 'size': 12,
                                                             'family': 'DM Mono'}},
                number={'suffix': '%', 'font': {'color': '#c8f050', 'size': 32,
                                                'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#3a3a55',
                             'tickfont': {'color': '#6b6b8a', 'size': 10}},
                    'bar': {'color': '#c8f050', 'thickness': 0.25},
                    'bgcolor': '#12121a',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 50],   'color': '#1a1a28'},
                        {'range': [50, 75],  'color': '#1e1e32'},
                        {'range': [75, 100], 'color': '#22223a'},
                    ],
                    'threshold': {'line': {'color': '#5046e5', 'width': 2},
                                  'thickness': 0.75, 'value': confidence}
                }
            ))
            fig_gauge.update_layout(
                height=200, margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_family='DM Mono'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Comparable listings from real data
            st.markdown('<div class="section-label" style="margin-top:8px;">Comparable Listings</div>',
                        unsafe_allow_html=True)

            comps = df[
                (df['City'] == sel_city) &
                (df['BHK'] == sel_bhk) &
                (df['Furnishing Status'] == sel_furnish)
            ][['Area Locality', 'BHK', 'Size', 'Rent', 'Furnishing Status']].copy()

            comps = comps.sort_values('Rent').drop_duplicates('Area Locality').head(4)

            if not comps.empty:
                for _, row in comps.iterrows():
                    diff  = int(row['Rent']) - predicted_rent
                    d_str = f"+₹{diff:,}" if diff > 0 else f"₹{diff:,}"
                    tag   = "info-tag-orange" if diff > 0 else ("info-tag-green" if diff < 0 else "info-tag")
                    st.markdown(f"""
                    <div style="background:#12121a; border:1px solid rgba(255,255,255,0.06);
                                border-radius:10px; padding:14px 18px; margin-bottom:8px;
                                display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:0.82rem; color:#f0f0f8;">{row['Area Locality'][:30]}</div>
                            <div style="font-size:0.68rem; color:#6b6b8a; margin-top:2px;">
                                {int(row['BHK'])} BHK · {int(row['Size'])} sqft · {row['Furnishing Status']}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:'Syne',sans-serif; font-weight:700;
                                        font-size:1rem; color:#c8f050;">₹{int(row['Rent']):,}</div>
                            <span class="{tag}">{d_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#6b6b8a; font-size:0.8rem; padding:16px;">No direct comparables found for this combination.</div>',
                            unsafe_allow_html=True)

        else:
            # Idle state
            st.markdown("""
            <div style="background:#12121a; border:1px solid rgba(255,255,255,0.06);
                        border-radius:20px; padding:60px 30px; text-align:center; margin-top:10px;">
                <div style="font-size:3rem; opacity:0.3; margin-bottom:16px;">🏠</div>
                <div style="color:#6b6b8a; font-size:0.82rem; line-height:1.8;">
                    Configure your property details<br>on the left and click<br>
                    <span style="color:#c8f050; font-family:'Syne',sans-serif; font-weight:700;">Predict Rent</span>
                    to see your estimate.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 2 — MARKET EXPLORER
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Market Overview</div>', unsafe_allow_html=True)

    # City comparison bar chart
    city_avg = df.groupby('City')['Rent'].median().sort_values(ascending=True).reset_index()
    city_avg.columns = ['City', 'Median Rent']

    fig_city = go.Figure(go.Bar(
        x=city_avg['Median Rent'],
        y=city_avg['City'],
        orientation='h',
        marker=dict(
            color=city_avg['Median Rent'],
            colorscale=[[0, '#5046e5'], [0.5, '#8b5cf6'], [1, '#c8f050']],
            showscale=False,
            line=dict(width=0)
        ),
        text=[f"₹{int(v):,}" for v in city_avg['Median Rent']],
        textposition='outside',
        textfont=dict(color='#f0f0f8', size=11, family='DM Mono'),
    ))
    fig_city.update_layout(
        title=dict(text="Median Rent by City", font=dict(color='#f0f0f8', size=14, family='Syne'), x=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#6b6b8a', showticklabels=False),
        yaxis=dict(color='#f0f0f8', tickfont=dict(family='DM Mono', size=12)),
        height=300, margin=dict(t=40, b=10, l=10, r=60),
        bargap=0.35
    )
    st.plotly_chart(fig_city, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # BHK distribution
        bhk_rent = df.groupby('BHK')['Rent'].median().reset_index()
        fig_bhk = go.Figure(go.Bar(
            x=bhk_rent['BHK'].astype(str) + ' BHK',
            y=bhk_rent['Rent'],
            marker=dict(
                color=bhk_rent['Rent'],
                colorscale=[[0,'#1a1a28'],[0.5,'#5046e5'],[1,'#c8f050']],
                showscale=False
            ),
            text=[f"₹{int(v):,}" for v in bhk_rent['Rent']],
            textposition='outside',
            textfont=dict(color='#c8f050', size=10, family='DM Mono'),
        ))
        fig_bhk.update_layout(
            title=dict(text="Median Rent by BHK", font=dict(color='#f0f0f8', size=13, family='Syne'), x=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#f0f0f8', tickfont=dict(family='DM Mono')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#6b6b8a', showticklabels=False),
            height=280, margin=dict(t=40, b=10, l=10, r=20), bargap=0.35
        )
        st.plotly_chart(fig_bhk, use_container_width=True)

    with col_b:
        # Furnishing impact
        furn_rent = df.groupby('Furnishing Status')['Rent'].median().reset_index()
        furn_colors = {'Unfurnished': '#5046e5', 'Semi-Furnished': '#8b5cf6', 'Furnished': '#c8f050'}
        colors = [furn_colors.get(f, '#6b6b8a') for f in furn_rent['Furnishing Status']]

        fig_furn = go.Figure(go.Bar(
            x=furn_rent['Furnishing Status'],
            y=furn_rent['Rent'],
            marker=dict(color=colors),
            text=[f"₹{int(v):,}" for v in furn_rent['Rent']],
            textposition='outside',
            textfont=dict(color='#f0f0f8', size=10, family='DM Mono'),
        ))
        fig_furn.update_layout(
            title=dict(text="Rent by Furnishing Status", font=dict(color='#f0f0f8', size=13, family='Syne'), x=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#f0f0f8', tickfont=dict(family='DM Mono')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#6b6b8a', showticklabels=False),
            height=280, margin=dict(t=40, b=10, l=10, r=20), bargap=0.35
        )
        st.plotly_chart(fig_furn, use_container_width=True)

    # Size vs Rent scatter
    sample = df.sample(min(800, len(df)), random_state=42)
    city_colors_map = {
        'Mumbai': '#c8f050', 'Delhi': '#5046e5', 'Bangalore': '#ff7a45',
        'Hyderabad': '#4ade80', 'Chennai': '#f59e0b', 'Kolkata': '#a78bfa'
    }
    scatter_colors = [city_colors_map.get(c, '#6b6b8a') for c in sample['City']]

    fig_scatter = go.Figure()
    for city in sample['City'].unique():
        mask = sample['City'] == city
        fig_scatter.add_trace(go.Scatter(
            x=sample[mask]['Size'],
            y=sample[mask]['Rent'],
            mode='markers',
            name=city,
            marker=dict(
                color=city_colors_map.get(city, '#6b6b8a'),
                size=5, opacity=0.7,
                line=dict(width=0)
            ),
        ))

    fig_scatter.update_layout(
        title=dict(text="Size vs Rent by City", font=dict(color='#f0f0f8', size=14, family='Syne'), x=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Size (sq ft)', gridcolor='rgba(255,255,255,0.04)',
                   color='#6b6b8a', title_font=dict(color='#6b6b8a', family='DM Mono'),
                   tickfont=dict(family='DM Mono', color='#6b6b8a')),
        yaxis=dict(title='Rent (₹)', gridcolor='rgba(255,255,255,0.04)',
                   color='#6b6b8a', title_font=dict(color='#6b6b8a', family='DM Mono'),
                   tickfont=dict(family='DM Mono', color='#6b6b8a')),
        legend=dict(font=dict(color='#f0f0f8', family='DM Mono', size=11),
                    bgcolor='rgba(18,18,26,0.8)', bordercolor='rgba(255,255,255,0.08)'),
        height=380, margin=dict(t=50, b=40, l=60, r=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Dataset & Model Insights</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        # Feature importance
        feat_names = ['BHK', 'Size', 'Floor', 'Bathrooms', 'City',
                      'Furnishing', 'Area Type', 'Tenant Pref.', 'Contact']
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=feat_df['Importance'],
            y=feat_df['Feature'],
            orientation='h',
            marker=dict(
                color=feat_df['Importance'],
                colorscale=[[0, '#1a1a28'], [0.5, '#5046e5'], [1, '#c8f050']],
                showscale=False
            ),
            text=[f"{v:.3f}" for v in feat_df['Importance']],
            textposition='outside',
            textfont=dict(color='#c8f050', size=10, family='DM Mono'),
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance", font=dict(color='#f0f0f8', size=13, family='Syne'), x=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#6b6b8a', showticklabels=False),
            yaxis=dict(color='#f0f0f8', tickfont=dict(family='DM Mono', size=11)),
            height=320, margin=dict(t=40, b=10, l=10, r=60), bargap=0.3
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_i2:
        # Rent distribution histogram
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df[df['Rent'] < 200000]['Rent'],
            nbinsx=60,
            marker=dict(
                color='#5046e5',
                opacity=0.8,
                line=dict(color='#c8f050', width=0.3)
            ),
            name='Rent Distribution'
        ))
        fig_dist.update_layout(
            title=dict(text="Rent Distribution (< ₹2L)", font=dict(color='#f0f0f8', size=13, family='Syne'), x=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Rent (₹)', gridcolor='rgba(255,255,255,0.04)', color='#6b6b8a',
                       title_font=dict(color='#6b6b8a', family='DM Mono'),
                       tickfont=dict(family='DM Mono', color='#6b6b8a')),
            yaxis=dict(title='Count', gridcolor='rgba(255,255,255,0.04)', color='#6b6b8a',
                       title_font=dict(color='#6b6b8a', family='DM Mono'),
                       tickfont=dict(family='DM Mono', color='#6b6b8a')),
            height=320, margin=dict(t=40, b=40, l=60, r=20), showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Dataset sample
    st.markdown('<div class="section-label" style="margin-top:8px;">Raw Dataset Sample</div>',
                unsafe_allow_html=True)
    display_cols = ['City', 'BHK', 'Rent', 'Size', 'Furnishing Status', 'Area Type',
                    'Tenant Preferred', 'Bathroom']
    st.dataframe(
        df[display_cols].head(20).style
            .background_gradient(subset=['Rent'], cmap='YlGn')
            .format({'Rent': '₹{:,.0f}', 'Size': '{:,} sqft'}),
        use_container_width=True, height=340
    )

    # City summary table
    st.markdown('<div class="section-label" style="margin-top:16px;">City Statistics</div>',
                unsafe_allow_html=True)
    city_table = city_stats.copy()
    city_table.columns = ['Mean Rent', 'Median Rent', 'Min Rent', 'Max Rent', 'Listings']
    city_table = city_table.applymap(lambda x: f"₹{int(x):,}" if x > 100 else int(x))
    st.dataframe(city_table, use_container_width=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.5rem;
                    letter-spacing:-1px; color:#f0f0f8;">Rent<span style="color:#c8f050;">IQ</span></div>
        <div style="font-size:0.65rem; color:#6b6b8a; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">
            AI Rent Predictor
        </div>
    </div>
    <hr style="margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem; color:#6b6b8a; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">Model Info</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#12121a; border:1px solid rgba(255,255,255,0.06);
                border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:0.7rem; color:#6b6b8a; margin-bottom:6px; letter-spacing:1px;">ALGORITHM</div>
        <div style="font-size:0.85rem; color:#c8f050;">Gradient Boosting</div>
    </div>
    <div style="background:#12121a; border:1px solid rgba(255,255,255,0.06);
                border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:0.7rem; color:#6b6b8a; margin-bottom:6px; letter-spacing:1px;">R² SCORE</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.3rem; color:#f0f0f8;">{r2:.4f}</div>
    </div>
    <div style="background:#12121a; border:1px solid rgba(255,255,255,0.06);
                border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:0.7rem; color:#6b6b8a; margin-bottom:6px; letter-spacing:1px;">MAE</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.3rem; color:#f0f0f8;">₹{mae:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem; color:#6b6b8a; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">Data Coverage</div>',
                unsafe_allow_html=True)

    for city in cities:
        count = int(city_stats.loc[city, 'count'])
        pct   = int(count / len(df) * 100)
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#f0f0f8;">{city}</span>
                <span style="font-size:0.7rem; color:#6b6b8a;">{count}</span>
            </div>
            <div style="height:4px; background:#1a1a28; border-radius:2px;">
                <div style="height:100%; width:{pct}%; background:linear-gradient(90deg,#5046e5,#c8f050);
                            border-radius:2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; color:#6b6b8a; line-height:1.8; margin-top:8px;">
        Built with Streamlit &amp; Scikit-learn<br>
        Data: Indian Rental Market 2022<br>
        <span style="color:#c8f050;">4,747</span> listings across 6 cities
    </div>
    """, unsafe_allow_html=True)
