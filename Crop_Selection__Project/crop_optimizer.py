
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------
# 1. PROJECT SETTINGS
# -----------------------------------------

np.random.seed(42)

districts = [
    "Beed",
    "Latur",
    "Dharashiv",
    "Nanded",
    "Parbhani",
    "Hingoli",
    "Jalna",
    "Chhatrapati Sambhajinagar"
]

crops = {
    "Soybean": {
        "family": "Legume",
        "base_yield": 13,
        "water_need": 650,
        "price": 4800,
        "cost": 30000
    },
    "Jowar": {
        "family": "Cereal",
        "base_yield": 12,
        "water_need": 450,
        "price": 3200,
        "cost": 22000
    },
    "Bajra": {
        "family": "Cereal",
        "base_yield": 11,
        "water_need": 400,
        "price": 2500,
        "cost": 19000
    },
    "Tur": {
        "family": "Pulse",
        "base_yield": 8,
        "water_need": 600,
        "price": 7000,
        "cost": 25000
    },
    "Cotton": {
        "family": "Commercial",
        "base_yield": 10,
        "water_need": 750,
        "price": 7000,
        "cost": 42000
    },
    "Gram": {
        "family": "Pulse",
        "base_yield": 9,
        "water_need": 350,
        "price": 5500,
        "cost": 22000
    }
}

# -----------------------------------------
# 2. GENERATE SAMPLE DATASET
# -----------------------------------------

records = []

for district in districts:
    for year in range(2004, 2024):

        rainfall = np.random.normal(650, 130)
        rainfall = max(250, rainfall)

        temperature = np.random.normal(29, 2.5)
        soil_ph = np.random.normal(7.2, 0.5)
        organic_carbon = np.random.uniform(0.25, 0.9)
        soil_moisture = np.random.uniform(15, 45)

        for crop, details in crops.items():

            water_need = details["water_need"]

            rainfall_effect = 1 - (
                abs(rainfall - water_need) / (water_need * 2)
            )

            rainfall_effect = max(0.35, min(1.15, rainfall_effect))

            soil_effect = 1 - abs(soil_ph - 7) * 0.08
            soil_effect = max(0.7, soil_effect)

            moisture_effect = soil_moisture / 30
            moisture_effect = max(0.65, min(1.25, moisture_effect))

            temperature_effect = 1 - abs(temperature - 28) * 0.025
            temperature_effect = max(0.7, temperature_effect)

            yield_value = (
                details["base_yield"]
                * rainfall_effect
                * soil_effect
                * moisture_effect
                * temperature_effect
                + np.random.normal(0, 1.2)
            )

            yield_value = max(2, yield_value)

            market_price = details["price"] * np.random.uniform(
                0.85, 1.15
            )

            records.append({
                "District": district,
                "Year": year,
                "Crop": crop,
                "Rainfall_mm": round(rainfall, 2),
                "Temperature_C": round(temperature, 2),
                "Soil_pH": round(soil_ph, 2),
                "Organic_Carbon": round(organic_carbon, 2),
                "Soil_Moisture": round(soil_moisture, 2),
                "Market_Price": round(market_price, 2),
                "Yield_Quintal_Hectare": round(yield_value, 2)
            })

df = pd.DataFrame(records)

df.to_csv("crop_dataset_synthetic.csv", index=False)

print("\nSample dataset created successfully!")
print(df.head())

# -----------------------------------------
# 3. PREPARE DATA FOR MACHINE LEARNING
# -----------------------------------------

features = [
    "District",
    "Crop",
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH",
    "Organic_Carbon",
    "Soil_Moisture"
]

target = "Yield_Quintal_Hectare"

X = df[features]
y = df[target]

categorical_features = ["District", "Crop"]

numerical_features = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH",
    "Organic_Carbon",
    "Soil_Moisture"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numbers",
            "passthrough",
            numerical_features
        )
    ]
)

# -----------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------------------
# 5. TRAIN RANDOM FOREST MODEL
# -----------------------------------------

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# -----------------------------------------
# 6. EVALUATE THE MODEL
# -----------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("---------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R-squared Score:", round(r2, 3))

# -----------------------------------------
# 7. SELECT DISTRICT FOR ANALYSIS
# -----------------------------------------

selected_district = "Beed"

latest_year = df["Year"].max()

latest_data = df[
    (df["District"] == selected_district)
    & (df["Year"] == latest_year)
].copy()

# -----------------------------------------
# 8. PREDICT CROP YIELD
# -----------------------------------------

latest_data["Predicted_Yield"] = model.predict(
    latest_data[features]
)

# -----------------------------------------
# 9. CALCULATE ESTIMATED REVENUE
# -----------------------------------------

latest_data["Estimated_Revenue"] = (
    latest_data["Predicted_Yield"]
    * latest_data["Market_Price"]
)

latest_data["Estimated_Profit"] = (
    latest_data["Estimated_Revenue"]
    - latest_data["Crop"].map(
        lambda crop: crops[crop]["cost"]
    )
)

# -----------------------------------------
# 10. RANK CROPS
# -----------------------------------------

latest_data = latest_data.sort_values(
    by="Estimated_Profit",
    ascending=False
)

latest_data["Rank"] = range(
    1, len(latest_data) + 1
)

print("\nCROP RECOMMENDATIONS")
print("---------------------------")

print(
    latest_data[
        [
            "Crop",
            "Predicted_Yield",
            "Market_Price",
            "Estimated_Revenue",
            "Estimated_Profit",
            "Rank"
        ]
    ].to_string(index=False)
)

# -----------------------------------------
# 11. SUGGEST CROP ROTATION
# -----------------------------------------

top_crops = latest_data["Crop"].tolist()

rotation = None

for first_crop in top_crops:
    for second_crop in top_crops:
        if first_crop != second_crop:
            if (
                crops[first_crop]["family"]
                != crops[second_crop]["family"]
            ):
                rotation = (first_crop, second_crop)
                break

    if rotation:
        break

print("\nSUGGESTED CROP ROTATION")
print("---------------------------")

if rotation:
    print("First crop:", rotation[0])
    print("Next crop:", rotation[1])
else:
    print("No suitable pair found.")

# -----------------------------------------
# 12. SAVE RESULTS
# -----------------------------------------

latest_data.to_csv(
    "crop_recommendation_results.csv",
    index=False
)

print("\nResults saved successfully!")

# -----------------------------------------
# 13. CREATE CROP YIELD CHART
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    latest_data["Crop"],
    latest_data["Predicted_Yield"]
)

plt.title(
    "Predicted Crop Yield - " + selected_district
)

plt.xlabel("Crop")
plt.ylabel("Predicted Yield (Quintal per Hectare)")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("crop_yield_chart.png")
plt.show()

# -----------------------------------------
# 14. CREATE PROFIT COMPARISON CHART
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    latest_data["Crop"],
    latest_data["Estimated_Profit"]
)

plt.title(
    "Estimated Crop Profit - " + selected_district
)

plt.xlabel("Crop")
plt.ylabel("Estimated Profit (Rs. per Hectare)")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("crop_profit_chart.png")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
