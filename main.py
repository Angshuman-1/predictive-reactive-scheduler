import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ortools.sat.python import cp_model
import collections
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# DATA INGESTION UTILITIES
# ==========================================
class DataLoader:
    @staticmethod
    def load_machines(filepath="machines.csv"):
        print(f"Reading machine data from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}. Please create it.")
        df = pd.read_csv(filepath)
        machines_dict = {}
        for _, row in df.iterrows():
            machines_dict[int(row['machine_id'])] = {
                'age': float(row['age_days']),
                'hours': float(row['hours_run'])
            }
        return machines_dict

    @staticmethod
    def load_jobs(filepath="jobs.csv"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}. Please create it.")
        df = pd.read_csv(filepath)
        df = df.sort_values(by=['job_id', 'sequence'])
        
        jobs_dict = collections.defaultdict(list)
        for _, row in df.iterrows():
            job_id = int(row['job_id'])
            machine_id = int(row['machine_id'])
            duration = int(row['duration'])
            complexity = float(row['complexity'])
            jobs_dict[job_id].append({'machine': machine_id, 'duration': duration, 'complexity': complexity})
            
        return [jobs_dict[i] for i in sorted(jobs_dict.keys())]

    @staticmethod
    def load_disruption(filepath="disruption.csv"):
        if not os.path.exists(filepath):
            return None
        df = pd.read_csv(filepath)
        row = df.iloc[0]
        return (int(row['machine_id']), int(row['start_time']), int(row['duration']))

# ==========================================
# PHASE 1: Machine Learning Breakdown Predictor
# ==========================================
class BreakdownPredictor:
    def __init__(self, history_filepath="history.csv"):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        self.history_filepath = history_filepath

    def train(self):
        if not os.path.exists(self.history_filepath):
            raise FileNotFoundError(f"Missing {self.history_filepath}. Run the generator script first.")
        
        print("\n[ML Engine] Training ML model...")
        df = pd.read_csv(self.history_filepath)
        X = df[['machine_id', 'machine_age_days', 'current_job_complexity', 'consecutive_hours_run']]
        y = df['downtime_duration']
        self.model.fit(X, y)
        self.is_trained = True
        print("[ML Engine] Training done.")

    def predict_buffer(self, machine_id, age, complexity, hours_run):
        if not self.is_trained:
            self.train()
        X_new = pd.DataFrame([[machine_id, age, complexity, hours_run]], 
                             columns=['machine_id', 'machine_age_days', 'current_job_complexity', 'consecutive_hours_run'])
        predicted_downtime = self.model.predict(X_new)[0]
        risk_factor = 1.0 
        return int(predicted_downtime * risk_factor)

# ==========================================
# PHASE 2 & 3: The Shop Floor Environment & Rescheduler
# ==========================================
class ShopFloorScheduler:
    def __init__(self, jobs_data, machines_data):
        self.jobs_data = jobs_data
        self.machines_data = machines_data
        self.num_machines = len(machines_data)
        self.predictor = BreakdownPredictor()

    def build_and_solve(self, use_ml_buffers=False, breakdown_event=None, enforce_sequence_from=None):
        model = cp_model.CpModel()
        horizon = sum(task['duration'] for job in self.jobs_data for task in job) * 3 
        
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        if use_ml_buffers:
            print("\nCalculating TARGETED ML predicted buffers...")
            
        bottleneck_machines = {1}

        # 1. Create Variables and Buffers
        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine = task['machine']
                duration = task['duration']
                complexity = task['complexity']
                
                # TARGETED BUFFER LOGIC
                if use_ml_buffers:
                    is_bottleneck = machine in bottleneck_machines
                    is_complex = complexity >= 4.0
                    
                    if is_bottleneck or is_complex:
                        m_data = self.machines_data[machine]
                        buffer = self.predictor.predict_buffer(
                            machine_id=machine, 
                            age=m_data['age'], 
                            complexity=complexity, 
                            hours_run=m_data['hours']
                        )
                        print(f"  [+] Buffer Applied: Job {job_id} on M{machine} -> {buffer} units")
                    else:
                        buffer = 0
                else:
                    buffer = 0

                suffix = f'_{job_id}_{task_id}'
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                
                task_interval = model.NewIntervalVar(start_var, duration, end_var, 'task_int' + suffix)
                all_tasks[job_id, task_id] = {'start': start_var, 'end': end_var, 'machine': machine}

                if buffer > 0:
                    buffered_end = model.NewIntVar(0, horizon, 'buf_end' + suffix)
                    machine_interval = model.NewIntervalVar(start_var, duration + buffer, buffered_end, 'mach_int' + suffix)
                    machine_to_intervals[machine].append(machine_interval)
                else:
                    machine_to_intervals[machine].append(task_interval)

        # 2. Add Precedence Constraints
        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1]['start'] >= all_tasks[job_id, task_id]['end'])

        # 3. Handle Breakdowns and Overlaps
        for machine in range(self.num_machines):
            intervals = machine_to_intervals[machine]
            if breakdown_event and breakdown_event[0] == machine:
                b_start, b_duration = breakdown_event[1], breakdown_event[2]
                b_end = b_start + b_duration
                b_interval = model.NewIntervalVar(b_start, b_duration, b_end, f'breakdown_M{machine}')
                intervals.append(b_interval)
            model.AddNoOverlap(intervals)

        # 4. Right-Shift Simulation (Locking sequence AND minimum start times)
        if enforce_sequence_from:
            for machine, tasks in enforce_sequence_from.items():
                for i in range(len(tasks)):
                    curr_task = tasks[i]
                    if 'job' not in curr_task: continue
                    
                    # Prevent the Compression Bug: Tasks cannot start earlier than originally planned!
                    orig_start = curr_task['start']
                    model.Add(all_tasks[curr_task['job'], curr_task['task']]['start'] >= orig_start)
                    
                    # Enforce the original machine sequence
                    if i < len(tasks) - 1:
                        next_task = tasks[i+1]
                        if 'job' in next_task:
                            model.Add(all_tasks[next_task['job'], next_task['task']]['start'] >= 
                                      all_tasks[curr_task['job'], curr_task['task']]['end'])

        # 5. Objective
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[job_id, len(job) - 1]['end'] for job_id, job in enumerate(self.jobs_data)])
        model.Minimize(obj_var)

        # 6. Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = self._extract_schedule(solver, all_tasks)
            if breakdown_event:
                schedule[breakdown_event[0]].append({
                    'type': 'BREAKDOWN', 'start': breakdown_event[1], 'end': breakdown_event[1] + breakdown_event[2]
                })
                schedule[breakdown_event[0]].sort(key=lambda x: x['start'])
            return {'makespan': solver.ObjectiveValue(), 'schedule': schedule}
        return None

    def _extract_schedule(self, solver, all_tasks):
        schedule = collections.defaultdict(list)
        for (job_id, task_id), task_vars in all_tasks.items():
            schedule[task_vars['machine']].append({
                'job': job_id, 'task': task_id, 
                'start': solver.Value(task_vars['start']), 'end': solver.Value(task_vars['end'])
            })
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x['start'])
        return schedule

    # ==========================================
    # PHASE 4: Cost Function Evaluation
    # ==========================================
    def evaluate_performance(self, schedule_dict, makespan, reference_schedule=None):
        """
        Calculates the real-world business cost of a schedule.
        """
        # Baseline cost to keep lights on and machines powered
        operating_cost_per_unit = 100 
        
        # The true cost of nervousness: setup tear-downs, WIP movement, operator confusion
        chaos_penalty_per_task = 3500 

        makespan_cost = makespan * operating_cost_per_unit
        instability_count = 0

        # If a disruption happened, compare the new start times against the original predictive plan
        if reference_schedule:
            orig_starts = {}
            for m, tasks in reference_schedule.items():
                for t in tasks:
                    if 'job' in t:
                        orig_starts[(t['job'], t['task'])] = t['start']

            for m, tasks in schedule_dict.items():
                for t in tasks:
                    if 'job' in t:
                        # If the task start time deviates from the original plan, trigger the massive penalty
                        if orig_starts.get((t['job'], t['task'])) != t['start']:
                            instability_count += 1

        chaos_cost = instability_count * chaos_penalty_per_task
        total_cost = makespan_cost + chaos_cost

        return total_cost, instability_count

    # ==========================================
    # PHASE 5: Plotly Gantt Chart Generator
    # ==========================================
    def visualize_gantt(self, schedule_dict, makespan, total_cost, title="Gantt Chart"):
        df_list = []
        base_time = datetime(2026, 1, 1, 0, 0, 0) 
        
        for machine, tasks in schedule_dict.items():
            for t in tasks:
                start_time = base_time + timedelta(hours=t['start'])
                end_time = base_time + timedelta(hours=t['end'])
                
                if 'job' in t:
                    task_name = f"Job {t['job']} (Task {t['task']})"
                    color_category = f"Job {t['job']}"
                else:
                    task_name = "MACHINE BREAKDOWN"
                    color_category = "Maintenance"

                df_list.append({
                    'Machine': f"Machine {machine}",
                    'Task': task_name,
                    'Start': start_time,
                    'Finish': end_time,
                    'Category': color_category,
                    'Duration': t['end'] - t['start']
                })

        df = pd.DataFrame(df_list)
        df = df.sort_values(by='Machine', ascending=False)

        # Inject Cost into the title
        full_title = f"{title} | Makespan: {makespan} | Total Cost: ${total_cost:,}"

        fig = px.timeline(
            df, x_start="Start", x_end="Finish", y="Machine", color="Category",
            hover_data={"Start": False, "Finish": False, "Duration": True, "Task": True},
            title=full_title,
            color_discrete_map={"Maintenance": "black", "Job 0": "#636efa", "Job 1": "#EF553B", "Job 2": "#00cc96"}
        )
        fig.layout.xaxis.type = 'date'
        fig.layout.xaxis.tickformat = '%H:%M' 
        fig.update_yaxes(autorange="reversed")
        
        filename = f"{title.replace(' ', '_').lower()}.html"
        fig.write_html(filename)
        print(f"--> Saved Visualization: {filename}")

# ==========================================
# EXECUTION & BENCHMARKING
# ==========================================
if __name__ == '__main__':
    try:
        print("\n--- Initializing Data ---")
        machines_data = DataLoader.load_machines("machines.csv")
        jobs_data = DataLoader.load_jobs("jobs.csv")
        disruption = DataLoader.load_disruption("disruption.csv")
        print("Successfully loaded configuration files.")
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    scheduler = ShopFloorScheduler(jobs_data, machines_data)

    print("\n--- 1. Baseline Predictive Schedule ---")
    baseline = scheduler.build_and_solve(use_ml_buffers=False)
    cost_base, _ = scheduler.evaluate_performance(baseline['schedule'], baseline['makespan'])
    print(f"Makespan: {baseline['makespan']} | Cost: ${cost_base:,}")
    scheduler.visualize_gantt(baseline['schedule'], baseline['makespan'], cost_base, title="1_Baseline_Schedule")

    print("\n--- 2. Robust Schedule (With ML Buffers) ---")
    robust = scheduler.build_and_solve(use_ml_buffers=True)
    cost_rob, _ = scheduler.evaluate_performance(robust['schedule'], robust['makespan'])
    print(f"Makespan: {robust['makespan']} | Cost: ${cost_rob:,}")
    scheduler.visualize_gantt(robust['schedule'], robust['makespan'], cost_rob, title="2_Robust_Schedule")

    if disruption:
        print(f"\n--- Simulating Disruption ---")
        print(f"** Machine {disruption[0]} fails at T={disruption[1]} for {disruption[2]} units. **")

        print("\n--- 3. Right-Shift Repair on BASELINE ---")
        rs_base = scheduler.build_and_solve(use_ml_buffers=False, breakdown_event=disruption, enforce_sequence_from=baseline['schedule'])
        cost_rs_base, inst_rs_base = scheduler.evaluate_performance(rs_base['schedule'], rs_base['makespan'], reference_schedule=baseline['schedule'])
        print(f"Makespan: {rs_base['makespan']} | Deviated Tasks: {inst_rs_base} | Total Cost: ${cost_rs_base:,}")
        scheduler.visualize_gantt(rs_base['schedule'], rs_base['makespan'], cost_rs_base, title="3_Right_Shift_Baseline")

        print("\n--- 4. Complete Regeneration Repair on BASELINE ---")
        regen_base = scheduler.build_and_solve(use_ml_buffers=False, breakdown_event=disruption)
        cost_regen_base, inst_regen_base = scheduler.evaluate_performance(regen_base['schedule'], regen_base['makespan'], reference_schedule=baseline['schedule'])
        print(f"Makespan: {regen_base['makespan']} | Deviated Tasks: {inst_regen_base} | Total Cost: ${cost_regen_base:,}")
        scheduler.visualize_gantt(regen_base['schedule'], regen_base['makespan'], cost_regen_base, title="4_Regeneration_Baseline")

        print("\n--- 5. Right-Shift Repair on ROBUST ---")
        rs_rob = scheduler.build_and_solve(use_ml_buffers=False, breakdown_event=disruption, enforce_sequence_from=robust['schedule'])
        cost_rs_rob, inst_rs_rob = scheduler.evaluate_performance(rs_rob['schedule'], rs_rob['makespan'], reference_schedule=robust['schedule'])
        print(f"Makespan: {rs_rob['makespan']} | Deviated Tasks: {inst_rs_rob} | Total Cost: ${cost_rs_rob:,}")
        scheduler.visualize_gantt(rs_rob['schedule'], rs_rob['makespan'], cost_rs_rob, title="5_Right_Shift_Robust")

        print("\n--- 6. Complete Regeneration Repair on ROBUST ---")
        regen_rob = scheduler.build_and_solve(use_ml_buffers=False, breakdown_event=disruption)
        cost_regen_rob, inst_regen_rob = scheduler.evaluate_performance(regen_rob['schedule'], regen_rob['makespan'], reference_schedule=robust['schedule'])
        print(f"Makespan: {regen_rob['makespan']} | Deviated Tasks: {inst_regen_rob} | Total Cost: ${cost_regen_rob:,}")
        scheduler.visualize_gantt(regen_rob['schedule'], regen_rob['makespan'], cost_regen_rob, title="6_Regeneration_Robust")
        
    print("\nSimulation complete. Open the generated HTML files in your browser.")
