# 🌱 EcoPackAI -- Sustainable Packaging Material Recommendation System

## 📌 Overview

**EcoPackAI** is an AI-powered system designed to recommend the most suitable, cost-effective, and eco-friendly packaging materials for physical product types. The system evaluates factors such as $CO_2$ emission footprint, unit cost efficiency, material durability (tensile strength), and capacity utilization to provide optimal recommendations.

This project integrates Machine Learning models, a high-performance **FastAPI** backend, PostgreSQL database persistence, and an interactive Bootstrap frontend UI into a complete end-to-end web application.

---

## 🚀 Features

- 🌿 **AI-Based Material Recommendation**: Uses XGBoost & Random Forest regressors for multi-objective material suitability scoring.
- 📉 **Environmental Impact Analysis**: Evaluates $CO_2$ impact scores balanced against material recyclability percentages.
- 💰 **Cost Efficiency Prediction**: Predicts normalized cost efficiency indices for candidate materials.
- 📊 **Smart Multi-Criteria Decision Making (MCDM)**: Ranks feasible materials based on weighted suitability scores ($40\% \text{ CO}_2 + 40\% \text{ Cost} + 20\% \text{ Capacity Utilization}$).
- ⚡ **FastAPI REST API**: Type-safe, high-performance API backend with Pydantic request validation.
- 📖 **Interactive Swagger & ReDoc API Docs**: Auto-generated interactive API documentation at `/docs` and `/redoc`.
- 🗄️ **PostgreSQL Database Integration**: Automatically creates and stores product submissions.
- 🎨 **Interactive Frontend UI**: Bootstrap 5 web app for instant recommendations.

---

## 🏗️ System Architecture

```
User (Browser) ──► FastAPI Backend (app.py) ──► ML Inference Engine (ecopackai.py) ──► PostgreSQL (db.py)
```

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic, Jinja2
- **Machine Learning**: Scikit-learn, XGBoost, Joblib
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL (`psycopg2-binary`)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript (Fetch API)

---

## 📂 Project Structure

```text
EcopackAI/
├── app.py                     # FastAPI web application & endpoints
├── ecopackai.py               # ML recommendation engine & model inference
├── db.py                      # PostgreSQL database connection & auto-table init
├── co2_model                  # Trained XGBoost regressor for CO2 index
├── cost_model                 # Trained Random Forest regressor for cost index
├── Ecopack-dataset.csv        # Packaging material dataset (~5,000 samples)
├── final_product_dataset.csv  # Product catalog dataset
├── Data_cleaning.ipynb        # Data exploration, feature engineering & model training notebook
├── test_api.py                # Automated API test script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Environment & build git exclusions
├── static/                    # Static CSS assets
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

3. **Configure PostgreSQL**:
   Update database credentials in `db.py` (or set environment variables):
   - Host: `localhost`
   - Database: `ecopackdb`
   - Default Port: `5432`

4. **Run FastAPI Web Server**:
   ```bash
   python app.py
   # OR via uvicorn directly:
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
| `GET` | `/` | Serves interactive web UI |
| `POST` | `/api/product` | Inserts new product into DB & returns created `product_id` |
| `GET` | `/api/products` | Retrieves all stored products from DB |
| `POST` | `/api/recommend` | Runs ML engine for a product and returns top recommended materials |
| `POST` | `/api/environment` | Looks up $CO_2$ impact score for a specific packaging material |

---

## 👨‍💻 Author

**Vaibhav Jain**

---

⭐ Star this repository if you found it useful!

