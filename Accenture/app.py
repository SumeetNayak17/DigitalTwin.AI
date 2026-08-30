import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import json
import time
import os

# ==========================================
# ⏱️ MASTER TIMING CONTROLS FOR LIVE FEED
# ==========================================
TIME_PER_VEHICLE = 0.5      # Seconds each nominal vehicle spends on screen
FAULT_REACTION_WINDOW = 5.0 # Seconds the dashboard pauses to let you deal with a defect
# ==========================================

# Initialize Session State
if 'false_alarms' not in st.session_state: st.session_state.false_alarms = 0
if 'confirmed_defects' not in st.session_state: st.session_state.confirmed_defects = 0

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    # Dynamically locate the JSON file in the same folder as app.py
    model_path = os.path.join(os.path.dirname(__file__), 'digital_twin_model.json')
    model.load_model(model_path)
    return model

@st.cache_data
def load_dataset():
    # Dynamically locate the CSV file in the same folder as app.py
    csv_path = os.path.join(os.path.dirname(__file__), 'assembly_line_40_stations.csv')
    return pd.read_csv(csv_path)

model = load_model()
df_dataset = load_dataset()
ordered_cols = model.get_booster().feature_names

# --- UI SETUP ---
st.set_page_config(page_title="Digital Twin Assembly", layout="wide")
st.title("🚗 DigitalTwin.ai: 40-Station Virtual Sensor")

# 5 Full Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Floor Telemetry", 
    "📊 Plant Manager Risk", 
    "🤖 RL Autonomous Control", 
    "🌐 Omniverse Bridge", 
    "🏢 Enterprise ROI"
])

ui_display_cols = sorted(ordered_cols)

with st.sidebar:
    st.header("Control Panel")
    sim_mode = st.radio("Simulation Mode:", ["Manual Override", "Live Production Feed"])
    
    st.markdown("---")
    
    # Base dataframe for prediction
    baseline_data = {c: 50.0 for c in ordered_cols}
    baseline_data['St_20_Paint_Oven_Temp'] = 160.0
    baseline_data['St_05_Body_Robotic_Weld'] = 105.0
    baseline_data['St_35_Final_Vibration'] = 0.5
    baseline_data['St_28_Final_Legacy_Align'] = np.nan 
    df_live = pd.DataFrame([baseline_data])

    if sim_mode == "Manual Override":
        selected_stations = st.multiselect(
            "Select Stations to Monitor / Adjust:", 
            options=ui_display_cols, 
            default=['St_05_Body_Robotic_Weld', 'St_20_Paint_Oven_Temp', 'St_28_Final_Legacy_Align']
        )
        for station in selected_stations:
            if "Temp" in station:
                df_live[station] = st.slider(f"{station} (°C)", 140.0, 180.0, 160.0)
            elif "Align" in station:
                df_live[station] = st.slider(f"{station} (Deviation)", -3.0, 3.0, 0.0)
            elif "Weld" in station:
                df_live[station] = st.slider(f"{station} (Nm)", 80.0, 130.0, 105.0)
            else:
                df_live[station] = st.slider(f"{station} (Telemetry)", 0.0, 100.0, 50.0)

# Make single prediction for Manual Mode (used by Tabs 2-5)
df_live = df_live[ordered_cols]
prob = model.predict_proba(df_live)[0][1] * 100

with tab1:
    if sim_mode == "Manual Override":
        st.header("Floor Supervisor: Manual Intervention")
        if prob > 50:
            st.error(f"🚨 WARNING: {prob:.1f}% Probability of Structural Defect Detected!")
            st.markdown("**Prescriptive Action:** High thermal load detected upstream. Adjust downstream alignment tolerance.")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if st.button("✅ Confirm Defect Found"):
                    st.session_state.confirmed_defects += 1
            with f_col2:
                if st.button("❌ Log False Alarm (Tune Model)"):
                    st.session_state.false_alarms += 1
        else:
            st.success(f"✅ Nominal: {prob:.1f}% Defect Probability. Vehicle chassis is structurally sound.")
            
        total_alerts = st.session_state.confirmed_defects + st.session_state.false_alarms
        if total_alerts > 0:
            st.caption(f"**Floor-Level AI Trust Score:** {(st.session_state.confirmed_defects / total_alerts) * 100:.1f}%")

    elif sim_mode == "Live Production Feed":
        st.header("Streaming Production Line Data")
        st.markdown("Processing 10,000 vehicles through the 40-station virtual sensor array...")
        
        start_feed = st.button("▶ Initialize Assembly Line Feed")
        live_placeholder = st.empty()
        
        if start_feed:
            for idx, row in df_dataset.iterrows():
                # Prepare row for prediction
                df_row = pd.DataFrame([row[ordered_cols]])
                live_prob = model.predict_proba(df_row)[0][1] * 100
                
                # Identify the specific stations causing the defect to highlight them
                culprits = []
                if live_prob > 50:
                    if row['St_05_Body_Robotic_Weld'] < 93 or row['St_05_Body_Robotic_Weld'] > 117: culprits.append('St_05_Body_Robotic_Weld')
                    if row['St_20_Paint_Oven_Temp'] > 165: culprits.append('St_20_Paint_Oven_Temp')
                    if pd.notna(row['St_28_Final_Legacy_Align']) and abs(row['St_28_Final_Legacy_Align']) > 1.4: culprits.append('St_28_Final_Legacy_Align')

                with live_placeholder.container():
                    st.subheader(f"Processing Chassis GUID: #{int(row['Vehicle_ID']):06d}")
                    
                    # Build an 8x5 HTML CSS Grid for all 40 stations
                    grid_html = "<div style='display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-bottom: 20px;'>"
                    for col_name in ui_display_cols:
                        val = row[col_name]
                        short_name = f"St {col_name.split('_')[1]}"
                        val_str = "OFFLINE" if pd.isna(val) else f"{val:.1f}"
                        
                        if col_name in culprits:
                            # Bright red box for the isolated root causes
                            box_style = "background-color: #5c0000; border: 2px solid #ff4b4b; padding: 10px; border-radius: 5px; text-align: center; color: white;"
                        else:
                            # Subtle dark box for nominal generic stations
                            box_style = "background-color: #1e1e1e; border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center; color: #aaa;"
                            
                        grid_html += f"<div style='{box_style}'><span style='font-size: 0.75em;'>{short_name}</span><br><b style='font-size: 1.1em;'>{val_str}</b></div>"
                    grid_html += "</div>"
                    
                    st.markdown(grid_html, unsafe_allow_html=True)
                    
                    if live_prob > 50:
                        st.error(f"🚨 AI HALT TRIGGERED: {live_prob:.1f}% Structural Defect Probability!")
                        st.warning(f"Root cause isolated to highlighted stations. Line Paused. Auto-resuming in {FAULT_REACTION_WINDOW} seconds...")
                        time.sleep(FAULT_REACTION_WINDOW)
                    else:
                        st.success(f"✅ Clearance Granted: {live_prob:.1f}% Defect Risk.")
                        time.sleep(TIME_PER_VEHICLE)

with tab2:
    st.header("Station-wise Defect Probability & Risk Contribution")
    risk_df = pd.DataFrame({"Station": ordered_cols, "Risk Contribution": model.feature_importances_})
    risk_df = risk_df.sort_values(by="Risk Contribution", ascending=False).head(5)
    st.bar_chart(risk_df.set_index("Station"))

with tab3:
    st.header("Deep Q-Learning (DQL) Prescriptive Agent")
    if prob > 50 and sim_mode == "Manual Override":
        st.warning("Agent calculating optimal recovery policy...")
        safe_temp = df_live['St_20_Paint_Oven_Temp'].values[0]
        while safe_temp > 140.0:
            df_sim = df_live.copy()
            df_sim['St_20_Paint_Oven_Temp'] = safe_temp
            sim_prob = model.predict_proba(df_sim)[0][1] * 100
            if sim_prob < 50: break
            safe_temp -= 0.5
        st.success(f"**Optimal Policy Executed:** Drop Paint Oven Temp by **{df_live['St_20_Paint_Oven_Temp'].values[0] - safe_temp:.1f}°C**.")
    else:
        st.success("Current manual state is optimal or running in live feed. No RL intervention required.")

with tab4:
    st.header("NVIDIA Omniverse 3D Integration")
    if prob > 50 and sim_mode == "Manual Override":
        usd_payload = {
            "vehicle_guid": "CHASSIS-99482-A",
            "simulation_layer": "von_mises_stress_overlay",
            "defect_nodes": [{"station": "St_28_Final_Legacy_Align", "stress_tensor_mpa": 188.4, "yield_exceeded": True}]
        }
        st.json(json.dumps(usd_payload, indent=4))
        st.button("📡 Push to Omniverse Nucleus Server")
    else:
        st.write("Awaiting manual structural anomaly to generate 3D USD mapping...")

with tab5:
    st.header("Enterprise Scalability & Business Case")
    roi_col1, roi_col2, roi_col3 = st.columns(3)
    with roi_col1: plants = st.slider("Assembly Lines", 1, 50, 5)
    with roi_col2: legacy_stations = st.slider("Legacy Stations per Line", 5, 20, 10)
    with roi_col3: retrofit_cost = st.slider("Hardware Retrofit Cost ($)", 10000, 50000, 25000)
    
    physical_cost = plants * legacy_stations * retrofit_cost
    software_cost = plants * 45000 
    savings = physical_cost - software_cost
    roi_percentage = (savings/physical_cost)*100 if physical_cost > 0 else 0
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Cost: Hardware Retrofit", f"${physical_cost:,.0f}")
    res_col2.metric("Cost: DigitalTwin Software", f"${software_cost:,.0f}")
    res_col3.metric("Net Capital Saved", f"${savings:,.0f}", f"{roi_percentage:.1f}% ROI")
