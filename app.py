# Importing Libraries

import numpy as np
import pandas as pd
import joblib
from flask import Flask , request , jsonify,render_template

from ecopackai import recommend_material
from ecopackai import recommend_materials_by_weight
from db import get_connection

# Dataset loading

product_df=pd.read_csv("final_product_dataset.csv")
material_df=pd.read_csv("Ecopack-dataset.csv")

# Model Loading

co2_model=joblib.load("co2_model")
cost_model=joblib.load("cost_model")


app = Flask(__name__)

#home page

@app.route("/")
def home():
    return render_template("index.html")




# 1. Product Input Handling API

@app.route("/api/product", methods=["POST"])
def product_input():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input provided"}), 400

    if "product_name" not in data or "weight_kg" not in data:
        return jsonify({"error": "product_name and weight_kg required"}), 400

    product_name = data["product_name"]
    weight = float(data["weight_kg"])

    # 🔽 Store in PostgreSQL
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO products (product_name, weight_kg) VALUES (%s, %s)",
        (product_name, weight)
    )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "message": "Product stored successfully in database",
        "product_name": product_name,
        "weight_kg": weight
    })

# Read products back from DB

@app.route("/api/products", methods=["GET"])
def get_products():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, product_name, weight_kg FROM products")
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

    return jsonify(products)


# -------------------------------
# 2. AI Material Recommendation API
# -------------------------------
@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "product_id" not in data:
        return jsonify({"error": "product_id is required"}), 400

    product_id = int(data["product_id"])

    #  Fetch product from PostgreSQL
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT product_name, weight_kg FROM products WHERE id = %s",
        (product_id,)
    )

    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return jsonify({"error": "Product not found"}), 404

    product_name, weight_kg = row

    #  Call ML using weight
    recommendations = recommend_materials_by_weight(weight_kg)

    return jsonify({
        "product_id": product_id,
        "product_name": product_name,
        "weight_kg": weight_kg,
        "recommendations": recommendations.to_dict(orient="records")
    })



# -------------------------------
# 3. Environmental Score API
# -------------------------------
@app.route("/api/environment", methods=["POST"])
def environment_score():
    data = request.get_json()

    if not data or "material" not in data:
        return jsonify({"error": "Material name is required"}), 400

    material = data["material"]

    row = material_df[material_df["Material_Type"] == material]

    if row.empty:
        return jsonify({"error": "Material not found"}), 404

    co2_score = float(row["CO2_Emission_Score"].values[0])

    return jsonify({
        "material": material,
        "co2_impact_score": co2_score
    })


if __name__ == "__main__":
    app.run(debug=True)








if __name__ == "__main__":
    app.run(debug=True)
