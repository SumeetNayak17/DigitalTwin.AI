# DigitalTwin.ai: Autonomous Virtual Sensor for Mixed-Model Assembly
**Accenture Innovation Challenge 2026 - Problem Track 4 Prototype**

🌐 **Live Prototype Deployment:** [Access the Live Streamlit App](YOUR_STREAMLIT_URL_HERE)

## Overview
This repository contains a functional Digital Twin prototype for a mixed-model vehicle assembly line, integrating principles of solid mechanics with machine learning. The solution addresses the challenge of uneven sensor coverage by deploying an XGBoost-driven "Virtual Sensor" and a Deep Q-Learning (DQL) prescriptive agent. This architecture bridges data gaps at legacy stations without requiring costly hardware retrofits or production downtime.

## Implementation Approach
Modern assembly lines possess rich telemetry at robotic welding and paint stations but suffer from "blind spots" at legacy equipment stations. This solution utilizes a machine learning surrogate model trained on early-stage station data to infer the physical state of the chassis at unmonitored legacy stations. 
* **Physics-Informed Defect Detection:** The algorithm isolates compounding, multi-causal variables—specifically thermal anomalies interacting with downstream mechanical misalignments—to predict structural defects before end-of-line testing.
* **Alert Fatigue Management:** Incorporates a continuous-learning human-in-the-loop feedback system, allowing floor supervisors to validate claims or log false alarms to recalibrate model sensitivity.
* **Prescriptive Autonomous Control:** An embedded RL agent dynamically calculates the exact operational adjustments required to return an anomalous chassis to a nominal state.

## Solution Architecture
* **Data Generation (`dataset_40.py`):** A physics-informed Python script simulating 10,000 vehicles across a continuous 40-station line (Body Construction, Paint Shop, Final Assembly). It injects realistic mechanical variances and intentionally masks 35% of legacy station data to simulate real-world sensor dropouts.
* **Predictive Engine (`training.py`):** An XGBoost Classifier that natively handles missing tabular data, achieving zero-latency edge inference to predict downstream defects based strictly on upstream telemetry.
* **Interactive UI (`app.py`):** A Streamlit dashboard providing distinct, role-based visualizations: live production feeds with CSS grid telemetry, Deep Q-Learning autonomous recovery, Omniverse USD integration, and enterprise ROI calculators.

## Dependencies & Execution Instructions
If running locally, ensure Python 3.8+ is installed:
```bash
pip install -r requirements.txt
python dataset_40.py
python training.py
streamlit run app.py