# 💻 Laptop Price Prediction — ML Project

A complete, portfolio-ready end-to-end Machine Learning project that predicts laptop prices with a quality score and market recommendation.

---

## 🗂️ Project Structure

```
laptop_price_Predication/
├── laptop_price.csv          # Original dataset (1,303 rows)
├── laptop_augmented.csv      # Augmented dataset (2,200+ rows)
├── generate_data.py          # Dataset augmentation script
├── eda.py                    # EDA & visualisation script
├── train_model.py            # Model training + hyperparameter tuning
├── app.py                    # Streamlit web application
├── model.pkl                 # Saved best model pipeline
├── model_meta.json           # Model metadata
├── requirements.txt          # Python dependencies
└── plots/                    # EDA visualisation outputs
```

---

## ⚙️ Setup & Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Augment dataset to 2,200+ rows
python generate_data.py

# 3. Run EDA (optional — saves plots to ./plots/)
python eda.py

# 4. Train model & save model.pkl
python train_model.py

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## 🧠 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Strong performer, tuned via RandomizedSearchCV |
| Gradient Boosting | Strong performer, tuned via RandomizedSearchCV |

Best model is auto-selected by R² score and saved as `model.pkl`.

---

## 📊 Features Used

| Feature | Engineering |
|---|---|
| Company, TypeName, OpSys | OneHotEncoding |
| CPU | Brand (Intel/AMD), Tier (High/Mid/Entry), GHz extracted |
| GPU | Tier (High/Mid-High/Mid/Integrated) |
| RAM | Numeric GB extracted |
| Memory | SSD GB + HDD GB extracted separately |
| ScreenResolution | Category (4K/QHD/FHD/HD) |
| Weight | Numeric kg extracted |

---

## 🚀 Deploy on Streamlit Cloud (Free)

1. Push the project to a **GitHub** repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo and set `app.py` as the entry point
4. Add a `requirements.txt` (already included) — Streamlit Cloud auto-installs deps
5. **Important:** You must commit `model.pkl` and `model_meta.json` to the repo, or add a startup script in `app.py` that runs training if the file is missing

> **Tip:** If `model.pkl` is > 100MB, use [Git LFS](https://git-lfs.github.com/)

---

## 🚀 Deploy on Render (Free Tier)

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New → **Web Service**
3. Set:
   - **Build Command:** `pip install -r requirements.txt && python generate_data.py && python train_model.py`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Set environment: `Python 3.10+`

---

## 📈 Output

- **💰 Predicted Price** — in EUR and USD
- **⭐ Quality Score** — Low / Medium / High
- **🏷️ Recommendation** — Budget / Mid-range / Premium
- **📊 Price Breakdown** — contribution of each spec
