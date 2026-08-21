# 🌱 EcoPackAI -- Sustainable Packaging Material Recommendation System

## 📌 Overview

**EcoPackAI** is an end-to-end AI-powered system designed to recommend the most suitable, cost-effective, and eco-friendly packaging materials for physical products. The system evaluates carbon footprint ($CO_2$ impact), unit cost efficiency, material durability (tensile strength), moisture barrier protection, and capacity utilization to provide optimal recommendations tailored to both **product weight** and **product category**.

This project integrates Machine Learning regression models, a high-performance **FastAPI** backend, PostgreSQL database persistence, `.env` security configuration, and an interactive Bootstrap + Chart.js web application.

---

## 🚀 Features

- 🌿 **AI-Based Material Recommendation**: Uses XGBoost & Random Forest regressors for multi-objective material suitability scoring.
- 📦 **Category-Aware Sensitivity Filtering**: Tailors material recommendations based on product category requirements (*Electronics*, *Food & Perishables*, *Fragile Glassware*, *Clothing*, *Heavy Industrial*, *General*).
- ⚖️ **Dynamic Weight & Capacity Fit**: Applies optimal capacity utilization and over-packaging penalty scoring to reward exact material fits and penalize oversized packaging.
- 📊 **Visual Comparison Charts**: Interactive **Chart.js** grouped bar charts and doughnut charts comparing $CO_2$ footprint, cost efficiency, and suitability distribution.
- 📉 **Environmental Impact Analysis**: Evaluates $CO_2$ impact scores balanced against material recyclability percentages.
- 💰 **Cost Efficiency Prediction**: Predicts normalized cost efficiency indices for candidate materials.
- 🔒 **Secure `.env` Configuration**: Uses `python-dotenv` for database credentials to keep passwords out of source control.
- ⚡ **FastAPI REST API**: Type-safe, high-performance API backend with Pydantic request validation.
- 📖 **Interactive Swagger & ReDoc API Docs**: Auto-generated interactive API documentation at `/docs` and `/redoc`.
- 🗄️ **PostgreSQL Database Integration**: Automatically initializes schemas, migrates columns, and stores product submissions.
- 🎨 **Modern Web Interface**: Glassmorphism UI with responsive cards, rank badges, CO₂ savings pills, and quick test presets.

---

## 🏗️ System Architecture

```
User (Browser) ──► FastAPI Backend (app.py) ──► ML Inference Engine (ecopackai.py) ──► PostgreSQL (db.py via .env)
```

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic, Jinja2, `python-dotenv`
- **Machine Learning**: Scikit-learn, XGBoost, Joblib
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL (`psycopg2-binary`)
- **Frontend**: HTML5, CSS3, Bootstrap 5, Chart.js, JavaScript (Fetch API)

---

## 📂 Project Structure

```text
EcopackAI/
├── app.py                     # FastAPI web application & REST endpoints
├── ecopackai.py               # ML recommendation engine & category constraint logic
├── db.py                      # PostgreSQL connection helper & schema auto-initialization
├── co2_model                  # Trained XGBoost regressor for CO2 impact index
├── cost_model                 # Trained Random Forest regressor for cost efficiency index
├── Ecopack-dataset.csv        # Packaging material dataset (~10,000 samples across 12 materials)
├── final_product_dataset.csv  # Product catalog dataset
├── .env                       # Local environment variables (Git-ignored secrets)
├── .env.example               # Environment variables template for developers
├── Data_cleaning.ipynb        # Data exploration, feature engineering & model training notebook
├── test_api.py                # Automated API test script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Environment & build git exclusions
├── static/                    # Custom CSS styles & glassmorphism system
└── templates/                 # HTML UI templates (index.html)
```

---

## ⚙️ Setup & Installation Instructions

1. **Clone repository**:
   ```bash
   git clone https://github.com/Vaibhavjain22/EcopackAI.git
   cd EcopackAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your local PostgreSQL credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   DB_HOST=localhost
   DB_NAME=ecopackdb
   DB_USER=postgres
   DB_PASSWORD=your_postgres_password
   DB_PORT=5432
   ```

4. **Run FastAPI Web Server**:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 5000 --reload
   ```

5. **Access Application**:
   - Web App Interface: [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Interactive Swagger API Docs: [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)
   - ReDoc API Documentation: [http://127.0.0.1:5000/redoc](http://127.0.0.1:5000/redoc)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves interactive web UI dashboard |
| `GET` | `/api/categories` | Returns available product categories & constraint labels |
| `POST` | `/api/product` | Inserts new product & category into DB, returning `product_id` |
| `GET` | `/api/products` | Retrieves recent stored products from DB |
| `POST` | `/api/recommend` | Runs ML engine for a product and returns top category-aware recommendations |
| `POST` | `/api/environment` | Looks up $CO_2$ impact score for a specific packaging material |

---

## 👨‍💻 Author

**Vaibhav Jain**

---

⭐ Star this repository if you found it useful!


