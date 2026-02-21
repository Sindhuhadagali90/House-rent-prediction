# -----------------------------------------------------------
# RentIQ — House Rent Prediction (FINAL FIXED VERSION)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="RentIQ — House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>
/* ——— fonts —— */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f;
    color: #f0f0f8;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
}

/* ——— Hero header —— */
.hero-header {
    background: linear-gradient(135deg, rgba(200,240,80,0.08), rgba(80,70,229,0.08));
    border: 1px solid rgba(200,240,80,0.15);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    overflow: hidden;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
}
.hero-title span { color: #c8f050; }

/* ——— Result card —— */
.result-card {
    background: linear-gradient(135deg, rgba(200,240,80,0.06), rgba(80,70,229,0.06));
    border: 1px solid rgba(200,240,80,0.2);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
}
.result-rent {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #c8f050;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d14 !important;
}

/* Buttons */
.stButton > button {
    background: #c8f050 !important;
    color: #0a0a0f !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD & PREPARE DATA
# -----------------------------------------------------------
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("House_Rent_Dataset.csv")

    def parse_floor(f):
        f = str(f).lower()
        if "ground" in f:
            return 0
        try:
            return int(f.split()[0])
        except:
            return 1

    df["Floor_Num"] = df["Floor"].apply(parse_floor)

    # label encoders
    le_city = LabelEncoder()
    le_furnish = LabelEncoder()
    le_area = LabelEncoder()
    le_tenant = LabelEncoder()
    le_contact = LabelEncoder()

    df["City_enc"] = le_city.fit_transform(df["City"])
    df["Furnish_enc"] = le_furnish.fit_transform(df["Furnishing Status"])
    df["AreaType_enc"] = le_area.fit_transform(df["Area Type"])
    df["Tenant_enc"] = le_tenant.fit_transform(df["Tenant Preferred"])
    df["Contact_enc"] = le_contact.fit_transform(df["Point of Contact"])

    features = [
        "BHK", "Size", "Floor_Num", "Bathroom",
        "City_enc", "Furnish_enc", "AreaType_enc", "Tenant_enc", "Contact_enc"
    ]

    X = df[features]
    y = df["Rent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc = max(0, min(100, round(r2 * 100, 1)))

    encoders = {
        "city": le_city,
        "furnish": le_furnish,
        "area": le_area,
        "tenant": le_tenant,
        "contact": le_contact,
    }

    return df, model, encoders, features, mae, r2, acc


@st.cache_data
def get_city_stats(_df):
    return _df.groupby("City")["Rent"].agg(
        ["mean", "median", "min", "max", "count"]
    ).round(0)


# -----------------------------------------------------------
# LOAD
# -----------------------------------------------------------
with st.spinner("Loading model..."):
    df, model, encoders, features, mae, r2, acc = load_and_prepare()
    city_stats = get_city_stats(df)


# -----------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------
st.markdown("""
<div class="hero-header">
  <div class="hero-title">Rent<span>IQ</span></div>
  <div class="hero-title" style="font-size:1.8rem; color:#6b6b8a; font-weight:400;">
    House Rent Prediction
  </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Predict Rent", "📊 Market Explorer", "🔍 Data Insights"])


# -----------------------------------------------------------
# TAB 1 — PREDICT
# -----------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        cities = sorted(df["City"].unique())
        furnishing = sorted(df["Furnishing Status"].unique())
        area_types = sorted(df["Area Type"].unique())
        tenants = sorted(df["Tenant Preferred"].unique())
        contacts = sorted(df["Point of Contact"].unique())

        sel_city = st.selectbox("City", cities)
        sel_bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
        sel_bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])
        sel_floor = st.number_input("Floor", min_value=0, max_value=30, value=1)
        sel_size = st.slider("Size (sqft)", 100, 5000, 900)
        sel_furnish = st.selectbox("Furnishing", furnishing)
        sel_area = st.selectbox("Area Type", area_types)
        sel_tenant = st.selectbox("Tenant Preferred", tenants)
        sel_contact = st.selectbox("Point of Contact", contacts)

        predict_clicked = st.button("Predict Rent")

    with col2:
        if predict_clicked:
            # encode inputs
            city_enc = encoders["city"].transform([sel_city])[0]
            furnish_enc = encoders["furnish"].transform([sel_furnish])[0]
            area_enc = encoders["area"].transform([sel_area])[0]
            tenant_enc = encoders["tenant"].transform([sel_tenant])[0]
            contact_enc = encoders["contact"].transform([sel_contact])[0]

            input_df = pd.DataFrame([[
                sel_bhk, sel_size, sel_floor, sel_bath,
                city_enc, furnish_enc, area_enc, tenant_enc, contact_enc,
            ]], columns=features)

            predicted = int(model.predict(input_df)[0])

            rent_low = int(predicted * 0.82)
            rent_high = int(predicted * 1.18)

            st.markdown(f"""
            <div class="result-card">
              <div class="result-rent">₹{predicted:,}</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Min Estimate", f"₹{rent_low:,}")
            st.metric("Max Estimate", f"₹{rent_high:,}")


# -----------------------------------------------------------
# TAB 3 — RAW DATA (FIXED)
# -----------------------------------------------------------
with tab3:
    st.subheader("Dataset Sample")
    display_cols = [
        "City", "BHK", "Rent", "Size", "Furnishing Status",
        "Area Type", "Tenant Preferred", "Bathroom"
    ]

    # FIXED — removed .style
    st.dataframe(df[display_cols].head(20), use_container_width=True, height=340)

    st.subheader("City Statistics")
    st.dataframe(city_stats, use_container_width=True)
