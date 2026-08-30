import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 10,000 Vehicle IDs
n_vehicles = 10000
vehicle_ids = np.arange(1, n_vehicles + 1)

# Station 1: Robotic Weld Torque (Normal distribution around 105 Nm)
torque = np.random.normal(105, 5, n_vehicles)

# Station 2: Ambient Temp (Varies between 18C and 30C)
temp = np.random.uniform(18, 30, n_vehicles)

# Station 3: Adhesive Flow Rate (Normal distribution)
flow_rate = np.random.normal(50, 2, n_vehicles)

# Station 4: Legacy Alignment (Manual check, ideal is 0 offset)
alignment = np.random.normal(0, 1.5, n_vehicles)

# Station 5: Fastener Vibration (Downstream indicator)
vibration = np.random.normal(0.5, 0.1, n_vehicles) + (alignment * 0.05)

# Calculate Defect (The Ground Truth Physics)
# Defect triggered if Temp is high AND alignment is off
defect_prob = (temp > 26) & (np.abs(alignment) > 1.2)
defects = np.where(defect_prob, 1, 0)

# Add some random noise/multi-causal factors (5% random defects)
random_noise = np.random.choice([0, 1], size=n_vehicles, p=[0.95, 0.05])
final_defects = np.maximum(defects, random_noise)

# Create DataFrame
df = pd.DataFrame({
    'Vehicle_ID': vehicle_ids,
    'Station_1_Weld_Torque': torque,
    'Station_2_Ambient_Temp': temp,
    'Station_3_Adhesive_Flow': flow_rate,
    'Station_4_Legacy_Alignment': alignment,
    'Station_5_Vibration': vibration,
    'End_of_Line_Defect': final_defects
})

# Inject "Legacy Data Gap" (Drop 30% of Station 4 data)
missing_indices = np.random.choice(df.index, size=int(n_vehicles * 0.3), replace=False)
df.loc[missing_indices, 'Station_4_Legacy_Alignment'] = np.nan

# Export to CSV
df.to_csv('assembly_line_data.csv', index=False)
print("Synthetic dataset generated: 'assembly_line_data.csv'")
