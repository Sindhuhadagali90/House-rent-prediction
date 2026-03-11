<div align="center">

# 🏠 RentIQ — House Rent Prediction System

**An AI-powered web application that predicts residential house rent across 6 major Indian cities using machine learning.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit)](https://share.streamlit.io)

<br/>

> Enter property details → Get instant rent estimate → Explore market analytics

<br/>

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model](#-model)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

India's rental market serves over **4 crore urban households**, yet pricing remains opaque — driven by broker intuition rather than data. **RentIQ** bridges this gap by providing instant, data-driven rent estimates for properties across Mumbai, Delhi, Bangalore, Hyderabad, Chennai, and Kolkata.

The system is trained on **4,746 real listings** from 2022 using a **Gradient Boosting Regressor** that captures non-linear relationships between location, size, furnishing, BHK configuration, and rent. The interactive Streamlit UI makes predictions accessible to everyone — no technical background required.

---

## 🌐 Live Demo

🚀 **[Launch RentIQ on Streamlit Cloud →](https://house-rent-prediction-bjahcuvzj7zreirc7kuasw.streamlit.app/)**



---

## ✨ Features

### 🎯 Rent Predictor
- Input 9 property attributes: city, BHK, size, floor, furnishing, area type, bathrooms, tenant preference, contact type
- Instant ML prediction with **minimum and maximum estimate range (±18%)**
- Price per sq ft calculation
- City average comparison with percentage delta
- Annual cost projection
- **Confidence score gauge** (interactive Plotly chart)
- **Comparable listings** pulled directly from the real dataset

### 📊 Market Explorer
- **Median rent by city** — horizontal bar chart with gradient coloring
- **Rent by BHK** — column chart showing per-configuration premium
- **Rent by furnishing status** — three-way comparison
- **Size vs Rent scatter plot** — 800-sample color-coded by city

### 🔍 Data Insights
- **Feature importance** bar chart from the trained GBR model
- **Rent distribution histogram** (listings below ₹2L/month)
- **Raw dataset sample** — first 20 rows with formatted values
- **City statistics table** — mean, median, min, max, and listing count

### 📋 Sidebar
- Live model metrics: R² score, MAE
- Per-city data coverage bars
- Dataset and attribution info

---

## 📸 Screenshots

| Rent Predictor | Market Explorer |
|---|---|
| *Prediction form with result panel* | *City comparison and scatter charts* |

| Data Insights | Sidebar |
|---|---|
| *Feature importance and distribution* | *Model metrics and city coverage* |

> *(Add screenshots to the `assets/` folder and update the image paths above)*

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **Web Framework** | Streamlit 1.32+ | UI, state, deployment |
| **ML Model** | Scikit-learn GBR | Prediction engine |
| **Data Processing** | Pandas, NumPy | Feature engineering |
| **Visualization** | Plotly | Interactive charts |
| **Hosting** | Streamlit Cloud | Free public deployment |

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) |
| **Records** | 4,746 listings |
| **Period** | April – July 2022 |
| **Missing values** | None |
| **Cities** | Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata |

### Features

| Column | Type | Description |
|---|---|---|
| `BHK` | Integer | Bedrooms + Hall + Kitchen |
| `Rent` | Integer | Monthly rent in ₹ *(target)* |
| `Size` | Integer | Property size in sq ft |
| `Floor` | String | Floor number and total floors |
| `Area Type` | Categorical | Super Area / Carpet Area / Built Area |
| `City` | Categorical | City of the listing |
| `Furnishing Status` | Categorical | Furnished / Semi-Furnished / Unfurnished |
| `Tenant Preferred` | Categorical | Bachelors / Family / Bachelors+Family |
| `Bathroom` | Integer | Number of bathrooms |
| `Point of Contact` | Categorical | Owner / Agent / Builder |

### City-wise Rent Summary

| City | Median Rent | Mean Rent | Listings |
|---|---|---|---|
| 🏙️ Mumbai | ₹52,000 | ₹85,321 | 972 |
| 🏛️ Delhi | ₹17,000 | ₹29,462 | 605 |
| 🌿 Bangalore | ₹14,000 | ₹24,966 | 886 |
| 🌊 Chennai | ₹14,000 | ₹21,614 | 891 |
| 💎 Hyderabad | ₹14,000 | ₹20,555 | 868 |
| 🎨 Kolkata | ₹8,500 | ₹11,645 | 524 |

---

## 🧠 Model

### Algorithm: Gradient Boosting Regressor

Selected after comparing Linear Regression, Decision Tree, Random Forest, SVR — GBR achieved the best test-set performance.

### Hyperparameters

```python
GradientBoostingRegressor(
    n_estimators    = 200,
    learning_rate   = 0.08,
    max_depth       = 5,
    min_samples_split = 4,
    random_state    = 42
)
```

### Performance

| Metric | Value |
|---|---|
| R² Score (test) | > 0.85 |
| Mean Absolute Error | ~₹5,000–8,000 |
| Train / Test Split | 80% / 20% |
| Training Samples | 3,797 |
| Test Samples | 949 |

### Feature Importance (approximate)

```
City               ███████████████████░░░  ~35%   ← dominant factor
Size (sq ft)       ██████████████░░░░░░░░  ~28%
Furnishing Status  ███████░░░░░░░░░░░░░░░  ~14%
BHK                █████░░░░░░░░░░░░░░░░░  ~10%
Bathroom           ███░░░░░░░░░░░░░░░░░░░   ~6%
Floor Number       ██░░░░░░░░░░░░░░░░░░░░   ~4%
Area Type          █░░░░░░░░░░░░░░░░░░░░░   ~1%
Tenant Preferred   █░░░░░░░░░░░░░░░░░░░░░   ~1%
Point of Contact   █░░░░░░░░░░░░░░░░░░░░░   ~1%
```

> A furnished 2BHK in **Mumbai** can cost **6× more** than the same configuration in **Kolkata**.

---

## 📁 Project Structure

```
resume-classification/
├── app.py                          ← Main Streamlit application
├── Cleaned_Resumes.csv             ← Training dataset
├── requirements.txt                ← Python dependencies
├── README.md                       ← You are here
├── notebooks/                      ← Jupyter notebooks (EDA, experiments)
├── data/                           ← Raw and processed data files
├── screenshots/                    ← App screenshots
└── images/                         ← Images used in README / UI
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10 or higher**
- pip package manager
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/house-rent-prediction.git
cd house-rent-prediction

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The model trains automatically on first load (~3–6 seconds) and is cached for fast subsequent interactions.

### Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

---

## ☁️ Deployment

### Streamlit Cloud (Recommended — Free)

1. **Fork** this repository to your GitHub account
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
3. Click **"New app"** → select your forked repo
4. Set **Main file path:** `app.py`
5. Click **Deploy**

Streamlit Cloud automatically installs `requirements.txt` and serves the app publicly.

### Local Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t rentiq .
docker run -p 8501:8501 rentiq
```

---

## 🔮 Roadmap

- [ ] Integrate live rental APIs (MagicBricks, 99acres)
- [ ] Add neighbourhood-level prediction with target encoding
- [ ] XGBoost / LightGBM model comparison panel
- [ ] Folium map for location-based browsing
- [ ] Multi-year data for trend analysis
- [ ] Mobile-optimized layout
- [ ] Save and compare multiple predictions

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Quick contribution workflow
git checkout -b feature/your-feature-name
# make changes
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# open a Pull Request on GitHub
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset by [Sourav Banerjee](https://www.kaggle.com/iamsouravbanerjee) on Kaggle
- [Streamlit](https://streamlit.io) for the incredible app framework
- [Scikit-learn](https://scikit-learn.org) for the ML toolkit
- [Plotly](https://plotly.com) for beautiful interactive charts

---

<div align="center">

Made with ❤️ · [Report a Bug](../../issues/new?template=bug_report.md) · [Request a Feature](../../issues/new?template=feature_request.md)

⭐ **Star this repo if RentIQ helped you!** ⭐

</div>
