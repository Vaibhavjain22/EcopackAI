# Importing Libraries

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List

from ecopackai import recommend_material, recommend_materials_by_weight
from db import get_connection, init_db

# Initialize FastAPI Application
app = FastAPI(
    title="EcoPackAI",
    description="AI-powered Sustainable Packaging Material Recommendation API",
    version="1.0.0"
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dataset loading
product_df = pd.read_csv("final_product_dataset.csv")
material_df = pd.read_csv("Ecopack-dataset.csv")

# Model Loading
co2_model = joblib.load("co2_model")
cost_model = joblib.load("cost_model")


# Pydantic Request Models
class ProductInputSchema(BaseModel):
    product_name: str = Field(..., example="electronics")
    weight_kg: float = Field(..., gt=0, example=2.5)

class RecommendRequestSchema(BaseModel):
    product_id: int = Field(..., example=1)

class EnvironmentRequestSchema(BaseModel):
    material: str = Field(..., example="Mushroom Mycelium")


@app.on_event("startup")
def startup_db_check():
    init_db()


# Home Page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



# 1. Product Input Handling API
@app.post("/api/product")
def product_input(data: ProductInputSchema):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO products (product_name, weight_kg) VALUES (%s, %s) RETURNING id",
            (data.product_name, data.weight_kg)
        )

        product_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        return {
            "message": "Product stored successfully in database",
            "product_id": product_id,
            "product_name": data.product_name,
            "weight_kg": data.weight_kg
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database insertion failed: {str(e)}"
        )


# Read products back from DB
@app.get("/api/products")
def get_products():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT id, product_name, weight_kg FROM products ORDER BY id DESC")
        rows = cur.fetchall()

        cur.close()
        conn.close()

        products = []
        for row in rows:
            products.append({
                "id": row[0],
                "product_name": row[1],
                "weight_kg": row[2]
            })

        return products
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}"
        )


# -------------------------------
# 2. AI Material Recommendation API
# -------------------------------
@app.post("/api/recommend")
def recommend(data: RecommendRequestSchema):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT product_name, weight_kg FROM products WHERE id = %s",
            (data.product_id,)
        )

        row = cur.fetchone()
        cur.close()
        conn.close()

        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

        product_name, weight_kg = row

        # Call ML engine using weight
        recommendations = recommend_materials_by_weight(weight_kg)

        return {
            "product_id": data.product_id,
            "product_name": product_name,
            "weight_kg": weight_kg,
            "recommendations": recommendations.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {str(e)}"
        )


# -------------------------------
# 3. Environmental Score API
# -------------------------------
@app.post("/api/environment")
def environment_score(data: EnvironmentRequestSchema):
    row = material_df[material_df["Material_Type"] == data.material]

    if row.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    co2_score = float(row["CO2_Emission_Score"].values[0])

    return {
        "material": data.material,
        "co2_impact_score": co2_score
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)

