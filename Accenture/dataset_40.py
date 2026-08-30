import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_vehicles = 10000
df = pd.DataFrame({'Vehicle_ID': np.arange(1, n_vehicles + 1)})

print("Simulating 40-Station Assembly Line (Hybrid Multi-Causal Generation)...")

# 1. Generate 36 generic "noise" stations
for i in range(1, 41):
    if i not in [5, 20, 28, 35]: 
        df[f'St_{i:02d}_Generic_Sensor'] = np.random.normal(50, 5, n_vehicles)

# 2. Inject Critical Stations
df['St_05_Body_Robotic_Weld'] = np.random.normal(105, 5, n_vehicles)
df['St_20_Paint_Oven_Temp'] = np.random.normal(160, 4, n_vehicles)
df['St_28_Final_Legacy_Align'] = np.random.normal(0, 1.5, n_vehicles)
df['St_35_Final_Vibration'] = np.random.normal(0.5, 0.1, n_vehicles) + (np.abs(df['St_28_Final_Legacy_Align']) * 0.05)

# --- FAILURE PATHWAY A: Solid Mechanics (Physics-Informed) ---
E_modulus = 70e3            # MPa 
alpha_mismatch = 2.3e-5     # 1/°C 
T_nominal = 145.0           # °C 
K_stiffness = 65.0          # MPa/mm 
yield_strength = 185.0      # MPa 

delta_T = np.maximum(0, df['St_20_Paint_Oven_Temp'] - T_nominal)
sigma_thermal = E_modulus * alpha_mismatch * delta_T                    
sigma_mechanical = K_stiffness * np.abs(df['St_28_Final_Legacy_Align']) 
sigma_von_mises = np.sqrt(sigma_thermal**2 + sigma_mechanical**2 - (sigma_thermal * sigma_mechanical))

physics_defect = sigma_von_mises > yield_strength

# --- FAILURE PATHWAY B: Basic Mechanical Error (Weld Failure) ---
# A defect occurs if torque is severely under-torqued (< 93) or over-torqued (> 117)
weld_defect = (df['St_05_Body_Robotic_Weld'] < 93) | (df['St_05_Body_Robotic_Weld'] > 117)

# --- COMBINE ALL PATHWAYS & ADD NOISE ---
# Defect occurs if Physics fails OR Weld fails
combined_defects = np.where(physics_defect | weld_defect, 1, 0)

# Add 3% random environmental noise (Pathway C)
random_noise = np.random.choice([0, 1], size=n_vehicles, p=[0.97, 0.03])
df['End_of_Line_Defect'] = np.maximum(combined_defects, random_noise)

# --- THE SENSOR GAP ---
# Mask 35% of the legacy alignment data to create the enterprise blind spot
missing_indices = np.random.choice(df.index, size=int(n_vehicles * 0.35), replace=False)
df.loc[missing_indices, 'St_28_Final_Legacy_Align'] = np.nan

# Reorder columns
cols = [c for c in df.columns if c != 'End_of_Line_Defect'] + ['End_of_Line_Defect']
df = df[cols]

df.to_csv('assembly_line_40_stations.csv', index=False)
print("Success! Hybrid Physics-Informed & Mechanical dataset generated.")