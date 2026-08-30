import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the new 40-station data
print("Loading 40-station assembly line data...")
df = pd.read_csv('assembly_line_40_stations.csv')

# 2. Prepare Features (X) and Target (y)
X = df.drop(['Vehicle_ID', 'End_of_Line_Defect'], axis=1)
y = df['End_of_Line_Defect']

# 3. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training the massive 40-Station Digital Twin Model...")
model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    max_depth=5,  # Slightly deeper to handle 40 variables
    learning_rate=0.1
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- MODEL EVALUATION ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# 6. Feature Importance (The ultimate proof)
print("--- TOP 5 FEATURE IMPORTANCES (Filtering out the noise) ---")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort descending

for i in range(5):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# 7. Overwrite the saved model
model.save_model('digital_twin_model.json')
print("\nModel saved successfully as 'digital_twin_model.json'")