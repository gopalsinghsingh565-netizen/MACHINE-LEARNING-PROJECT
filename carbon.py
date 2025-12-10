import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Create or load dataset
# (For demo, we generate synthetic data)
data = {
    'Electricity_kWh': [120, 250, 310, 400, 180, 500, 320, 610, 700, 850],
    'Fuel_Litres': [20, 35, 50, 65, 25, 80, 55, 100, 120, 130],
    'Water_Litres': [800, 1200, 1500, 2000, 1000, 2500, 1700, 3000, 3300, 3500],
    'Distance_km': [15, 30, 50, 60, 25, 80, 55, 100, 110, 130],
}

df = pd.DataFrame(data)

# Step 3: Calculate carbon emissions using emission factors
# (Average emission factors)
EF_electricity = 0.82   # kg CO2 per kWh
EF_fuel = 2.31          # kg CO2 per litre
EF_water = 0.0003       # kg CO2 per litre
EF_distance = 0.12      # kg CO2 per km

df['Carbon_Emission(kgCO2)'] = (
    df['Electricity_kWh'] * EF_electricity +
    df['Fuel_Litres'] * EF_fuel +
    df['Water_Litres'] * EF_water +
    df['Distance_km'] * EF_distance
)

print("\n=== Dataset with Carbon Emissions ===")
print(df)

# Step 4: Split dataset for ML
X = df[['Electricity_kWh', 'Fuel_Litres', 'Water_Litres', 'Distance_km']]
y = df['Carbon_Emission(kgCO2)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict emissions
y_pred = model.predict(X_test)

# Step 7: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# Step 8: Visualize Actual vs Predicted emissions
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Carbon Emissions')
plt.xlabel('Actual Emissions (kg CO2)')
plt.ylabel('Predicted Emissions (kg CO2)')
plt.grid(True)
plt.show()

# Step 9: Predict for new input (future data)
print("\n=== Future Prediction Example ===")
new_data = pd.DataFrame({
    'Electricity_kWh': [600],
    'Fuel_Litres': [75],
    'Water_Litres': [2500],
    'Distance_km': [90]
})

predicted_emission = model.predict(new_data)
print(f"Predicted Carbon Emission: {predicted_emission[0]:.2f} kg CO₂")

# Step 10: Save model (optional)
import joblib
joblib.dump(model, 'carbon_footprint_model.pkl')
print("\nModel saved as carbon_footprint_model.pkl ✅")