import numpy as np
import pandas as pd

def generate_realistic_history(filename="history.csv", num_samples=1000):
    """
    Generates a realistic historical dataset for machine breakdowns using 
    Weibull and Lognormal probability distributions.
    """
    #Commented out to ensure fresh data generation on every execution
    #np.random.seed(42) # Seed for reproducibility of the training set
    
    # 1. Generate Machine Profiles
    machine_ids = np.random.randint(0, 3, num_samples)
    base_ages = {0: 100, 1: 800, 2: 1800}
    machine_age_days = np.array([base_ages[m] + np.random.uniform(-50, 200) for m in machine_ids])
    
    # 2. Generate Job Complexity
    current_job_complexity = np.random.normal(loc=3.0, scale=1.0, size=num_samples)
    current_job_complexity = np.clip(current_job_complexity, 1.0, 5.0)
    
    # 3. Simulate "Consecutive Hours Run" (Time Between Failures using Weibull)
    shape_k = 1.5 
    consecutive_hours_run = []
    for m_id in machine_ids:
        scale_lambda = 24 if m_id == 0 else (16 if m_id == 1 else 8)
        hours = scale_lambda * np.random.weibull(shape_k)
        consecutive_hours_run.append(hours)
    
    consecutive_hours_run = np.clip(consecutive_hours_run, 0.5, 48.0)

    # 4. Calculate Downtime Duration (Target Variable using Lognormal)
    mu, sigma = 1.2, 0.5 
    base_downtime = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    
    # Mathematical link for the ML model to learn
    age_penalty = 1 + (machine_age_days / 2000)      
    complexity_penalty = 1 + (current_job_complexity / 5) 
    
    final_downtime = base_downtime * age_penalty * complexity_penalty
    final_downtime = np.round(np.clip(final_downtime, 0.5, 50.0), 1)

    # 5. Package and Save
    df = pd.DataFrame({
        'machine_id': machine_ids,
        'machine_age_days': np.round(machine_age_days, 0).astype(int),
        'current_job_complexity': np.round(current_job_complexity, 2),
        'consecutive_hours_run': np.round(consecutive_hours_run, 1),
        'downtime_duration': final_downtime
    })
    
    df.to_csv(filename, index=False)
    print(f"--> Generated {num_samples} records in '{filename}'.")

def generate_disruption_event(filename="disruption.csv"):
    """
    Generates a single disruption event. The duration severity is scaled 
    to match the machine profiles established in the history dataset.
    """
    # No fixed seed here, so you get a different disruption every time you run it
    machine_id = np.random.choice([0, 1, 2])
    start_time = np.random.randint(5, 25)
    
    # Base duration logic matching the machine profiles
    if machine_id == 0:
        duration = np.random.randint(5, 12)  # New machine = quick fix
    elif machine_id == 1:
        duration = np.random.randint(10, 25) # Mid-life machine = moderate fix
    else:
        duration = np.random.randint(20, 45) # Old machine = major breakdown
        
    df = pd.DataFrame({
        'machine_id': [machine_id],
        'start_time': [start_time],
        'duration': [duration]
    })
    
    df.to_csv(filename, index=False)
    print(f"--> Generated 1 disruption event in '{filename}'.")
    print(f"    Event details: Machine {machine_id} fails at T={start_time} for {duration} units.")

if __name__ == "__main__":
    print("Generating Datasets for Simulator...")
    generate_realistic_history("history.csv", num_samples=1000)
    generate_disruption_event("disruption.csv")
    print("\nDone! You can now run main.py.")
