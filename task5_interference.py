import joblib
import pandas as pd

# 1. Load the pre-trained Random Forest model
# Ensure 'random_forest_model.pkl' is in the same folder as this script
model = joblib.load('random_forest_model.pkl')

print("Model successfully loaded into memory!\n")

# 2. Define the new hypothetical alloy (Task 5)
# Ensure the values sum to exactly 100.0%
hypothetical_alloy = {
    'Fe': 85.0,  # Base Iron
    'Ni': 2.0,
    'Co': 0.0,
    'Cr': 10.0,  # Chromium for corrosion/hardenability
    'Mn': 1.0,
    'C':  0.5,   # Carbon for interstitial strengthening
    'Mo': 0.5,
    'Si': 0.5,
    'Cu': 0.0,
    'Al': 0.0,
    'W':  0.0,
    'V':  0.0,
    'Ti': 0.5,   # Titanium for carbide formation
    'Nb': 0.0
}

# Convert the dictionary to a pandas DataFrame
# This ensures scikit-learn maps the values to the exact feature names it trained on
new_alloy_df = pd.DataFrame([hypothetical_alloy])

# 3. Run the Inference Engine
predicted_strength = model.predict(new_alloy_df)[0]

print("--- TASK 5: PREDICTION RESULTS ---")
print(f"Predicted Tensile Strength: {predicted_strength:.2f} psi")