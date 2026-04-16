# Predictive-Reactive Manufacturing Scheduling with ML

An operations research framework that pairs a Machine Learning predictive model with a Google OR-Tools Constraint Programming (CP-SAT) solver to generate robust job shop schedules and evaluate reactive repair policies.

## Overview

This project simulates a dynamic manufacturing environment subject to unexpected machine breakdowns. It demonstrates how inserting **Targeted ML Buffers** into a predictive schedule can absorb disruptions and drastically reduce the real-world financial "Cost of Chaos" compared to naive scheduling approaches.

The framework compares two reactive rescheduling policies after a disruption:
1. **Right-Shift Rescheduling:** Postponing remaining operations while locking the original sequence.
2. **Complete Regeneration:** Re-optimizing the entire remaining schedule from scratch.

## Key Features
* **Machine Learning Predictor:** A Random Forest model trained on historical Weibull/Lognormal breakdown data to predict downtime duration based on machine age and job complexity.
* **Targeted Buffering:** Strategically inserts idle time only on bottleneck machines and high-risk jobs to minimize the initial makespan penalty.
* **Total Cost Function:** Evaluates schedules not just on raw makespan, but by applying a heavy financial penalty for "schedule nervousness" (deviated tasks).
* **Interactive Visualizations:** Automatically generates Plotly HTML Gantt charts to visually analyze cascading delays vs. buffer absorption.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Angshuman-1/predictive-reactive-scheduler.git
cd predictive-reactive-scheduler
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Step 1: Generate the Environment Data**
Run the data generator to create realistic machine profiles, job complexities, and historical breakdown records.
```bash
python generate_data.py
```
*(Note: You can manually edit `jobs.csv`, `machines.csv`, and `disruption.csv` to test specific sequence traps and bottlenecks).*

**Step 2: Run the Simulation**
Execute the main scheduling engine.
```bash
python main.py
```

## Output
The simulation will output the mathematical makespan and Total Cost (Makespan Cost + Instability Penalty) in the terminal. It will also generate 6 interactive HTML Gantt charts comparing the Baseline and Robust schedules before and after the disruption.
