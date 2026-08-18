

# Importing Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Importing the dataset

material_df=pd.read_csv("Ecopack-dataset.csv")
product_df=pd.read_csv("final_product_dataset.csv")

material_df.head(5)

product_df.head(5)

"""# ***Data Cleaning***"""

material_df.drop_duplicates(inplace=True)
product_df.drop_duplicates(inplace=True)

product_df.fillna(product_df.mean(numeric_only=True),inplace=True)
material_df.fillna(material_df.mean(numeric_only=True), inplace=True )

product_df.isnull().sum()

material_df.isnull().sum()

"""## ***Feature Engineering***"""

#co2 Impact index

material_df['Co2_impact_index']=material_df['CO2_Emission_Score']*(1-material_df['Recyclability_Percent']/100)

material_df['Co2_impact_index']

#cost efficiency

material_df["Cost_Normalized"] = (
    material_df["Cost_per_unit"] - material_df["Cost_per_unit"].min()
) / (
    material_df["Cost_per_unit"].max() - material_df["Cost_per_unit"].min()
)

material_df["Cost_Efficiency_Index"] = 1 - material_df["Cost_Normalized"]



"""# ***ML dataset preparation***"""

FEATURES = [
    "Tensile_Strength_MPa",
    "Weight_Capacity_kg",
    "Moisture_Barrier_Grade",
    "Biodegradability_Score",
    "Recyclability_Percent"
]

X = material_df[FEATURES]
y_cost = material_df["Cost_Efficiency_Index"]
y_co2 = material_df["Co2_impact_index"]

X_train, X_test, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# ***Model Training***"""

# ---- Cost Model (Random Forest)
cost_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
cost_model.fit(X_train_scaled, y_cost_train)

# ---- CO2 Model (XGBoost)
co2_model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)
co2_model.fit(X_train_scaled, y_co2_train)

"""# ***Model Evaluation***"""

cost_pred=cost_model.predict(X_test_scaled)
co2_pred=co2_model.predict(X_test_scaled)

def accuracy():
    print("Cost Model R2:", r2_score(y_cost_test, cost_pred))
    print("Carbon Model R2:", r2_score(y_co2_test, co2_pred))

"""# ***Saving the models***"""

import joblib
joblib.dump(co2_model,"co2_model")
joblib.dump(cost_model,"cost_model")

"""# ***PRODUCT-AWARE RECOMMENDATION LOGIC***"""

product_df['product_weight_kg']=product_df['product_weight_g']/1000


def recommend_material(index):
    product=product_df.iloc[index]

    #extracting the feasable materials

    feasable_material=(material_df['Weight_Capacity_kg']>=product['product_weight_kg'])

    #making prediction

    feasable_material_df = material_df[feasable_material].copy()

    x=feasable_material_df[FEATURES]
    x_scaled=scaler.transform(x)

    #co2 prediction

    feasable_material_df['Co2_impact_index_pred']=co2_model.predict(x_scaled)

    #cost prediction

    feasable_material_df['cost_efficiency_pred']= cost_model.predict(x_scaled)

    # introducing a new feature for better recommendation of materials "capacity_utilization"

    feasable_material_df['capacity_utilization']=product['product_weight_kg']/feasable_material_df['Weight_Capacity_kg']

    # Normalising

    feasable_material_df['co2_norm']=(
        (feasable_material_df['Co2_impact_index_pred']- feasable_material_df['Co2_impact_index_pred'].min())/
        (feasable_material_df['Co2_impact_index_pred'].max()-feasable_material_df['Co2_impact_index_pred'].min())
    )

    feasable_material_df['cost_norm']=(
        (feasable_material_df['cost_efficiency_pred']-feasable_material_df['cost_efficiency_pred'].min())/
        (feasable_material_df['cost_efficiency_pred'].max()-feasable_material_df['cost_efficiency_pred'].min())
    )

    feasable_material_df['uitl_norm']=(
        (feasable_material_df['capacity_utilization']-feasable_material_df['capacity_utilization'].min())/
        (feasable_material_df['capacity_utilization'].max()-feasable_material_df['capacity_utilization'].min())
    )
    #final suitability score
    feasable_material_df['suitability_score'] = (
        (0.4 * (1 - feasable_material_df['co2_norm'])) +
        (0.4 * (1 - feasable_material_df['cost_norm']))+
        (0.2 * feasable_material_df['uitl_norm'])
    )

    """# ***Final Ranking***"""

    top_materials = (feasable_material_df.sort_values(
        by='suitability_score', ascending=False
    ).drop_duplicates(subset='Material_Type', keep='first').head(5)
    )

    
    print("Your input product is :" ,product_df.iloc[index,0])
    print("Best pakacaging materials according to your product is:")

    return top_materials[['Material_Type','Co2_impact_index_pred','cost_efficiency_pred','suitability_score']]

    

# recommending material by weight for storing them in DB

def recommend_materials_by_weight(weight_kg):
    feasible_mask = material_df['Weight_Capacity_kg'] >= weight_kg
    feasible_material_df = material_df[feasible_mask].copy()

    if feasible_material_df.empty:
        # Fallback to materials with maximum available capacity
        max_cap = material_df['Weight_Capacity_kg'].max()
        feasible_material_df = material_df[material_df['Weight_Capacity_kg'] == max_cap].copy()

    X = feasible_material_df[FEATURES]
    X_scaled = scaler.transform(X)

    feasible_material_df['Co2_impact_index_pred'] = co2_model.predict(X_scaled)
    feasible_material_df['cost_efficiency_pred'] = cost_model.predict(X_scaled)

    # Calculate Capacity Fit Ratio (weight_kg / material capacity)
    # Ratio is between 0 and 1 for feasible items
    feasible_material_df['capacity_ratio'] = weight_kg / feasible_material_df['Weight_Capacity_kg']

    # Apply Over-Packaging Penalty: Reward optimal fit (ratio ~ 0.5 - 1.0), penalize ratio < 0.2 (over-packaging)
    # Penalty multiplier decreases exponentially as material capacity becomes excessively oversized
    feasible_material_df['fit_score'] = (
        feasible_material_df['capacity_ratio'] * 
        np.exp(-1.5 * (1.0 - feasible_material_df['capacity_ratio']))
    )

    # Group by Material_Type to compute clean average predictions across material instances
    grouped = feasible_material_df.groupby('Material_Type').agg({
        'Co2_impact_index_pred': 'mean',
        'cost_efficiency_pred': 'mean',
        'fit_score': 'mean'
    }).reset_index()

    # Normalization across material types for current weight query
    co2_range = grouped['Co2_impact_index_pred'].max() - grouped['Co2_impact_index_pred'].min()
    co2_min = grouped['Co2_impact_index_pred'].min()
    grouped['co2_norm'] = (grouped['Co2_impact_index_pred'] - co2_min) / (co2_range if co2_range > 0 else 1.0)

    cost_range = grouped['cost_efficiency_pred'].max() - grouped['cost_efficiency_pred'].min()
    cost_min = grouped['cost_efficiency_pred'].min()
    grouped['cost_norm'] = (grouped['cost_efficiency_pred'] - cost_min) / (cost_range if cost_range > 0 else 1.0)

    fit_range = grouped['fit_score'].max() - grouped['fit_score'].min()
    fit_min = grouped['fit_score'].min()
    grouped['fit_norm'] = (grouped['fit_score'] - fit_min) / (fit_range if fit_range > 0 else 1.0)

    # Composite Suitability Score (35% CO2 reduction, 35% Cost efficiency, 30% Optimal capacity fit)
    grouped['suitability_score'] = (
        0.35 * (1.0 - grouped['co2_norm']) +
        0.35 * (1.0 - grouped['cost_norm']) +
        0.30 * grouped['fit_norm']
    )

    # Sort descending and return top 3
    top = grouped.sort_values('suitability_score', ascending=False).head(3)

    return top[['Material_Type',
                'Co2_impact_index_pred',
                'cost_efficiency_pred',
                'suitability_score']]


