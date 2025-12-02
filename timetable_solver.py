import sys
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from tabulate import tabulate
from ortools.sat.python import cp_model
import math

class TimetableSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self, limit: int):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._solution_limit = limit
        self._start_time = time.time()

    def on_solution_callback(self):
        current_time = time.time()
        print(f'Solution {self._solution_count} found in {current_time - self._start_time:.2f} s')
        self._solution_count += 1
        if self._solution_count >= self._solution_limit:
            print(f'Stopping search after finding {self._solution_limit} solutions.')
            self.StopSearch()

    def solution_count(self):
        return self._solution_count

class TimetableScheduler:
    def __init__(self, config_path='config.json', freq_path='frequencies.json', 
                 teacher_path='teachers.json', room_path='rooms.json'):
        self.config = self._load_json(config_path)
        self.frequencies = self._load_json(freq_path)
        self.teacher_assignments = self._load_json(teacher_path)
        self.room_assignments = self._load_json(room_path)
        
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.variables = {}
        
        self._process_config()
        self._prepare_solver_data()

    def _load_json(self, path: str) -> Any:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {path}: {e}")
            sys.exit(1)

    def _process_config(self):
        """Process raw config into usable class attributes."""
        self.divisions = self.config['divisions']
        self.raw_lectures = self.config['lectures'] # Just the names, e.g. ["DSA", "Math"]
        self.raw_labs = self.config['labs']
        self.days = self.config['days']
        
        self.start_hour = self.config['start_hour']
        self.end_hour = self.config['end_hour']
        self.lunch_start = self.config.get('lunch_start_hour')
        
        if self.start_hour >= self.end_hour:
            raise ValueError("Start hour must be before end hour")

        self.total_slots = self.end_hour - self.start_hour
        self.slots = list(range(self.total_slots))
        
        self.lunch_slot_idx = None
        if self.lunch_start and self.start_hour <= self.lunch_start < self.end_hour:
            self.lunch_slot_idx = self.lunch_start - self.start_hour
            print(f"Lunch configured at slot index: {self.lunch_slot_idx} ({self.lunch_start}:00)")

    def _prepare_solver_data(self):
        """
        Refactored: Instead of 1 ID per subject, we generate 1 ID per (Subject, Teacher) combo.
        """
        print("\n--- Preparing Data ---")
        # 1. Invert Teacher Map: Subject -> List[Teachers]
        self.subject_to_teachers = {s: [] for s in self.raw_lectures + self.raw_labs}
        self.teacher_unavailability = {}

        for teacher, data in self.teacher_assignments.items():
            classes = data if isinstance(data, list) else data.get('classes', [])
            
            # Store unavailability if present
            if isinstance(data, dict) and 'unavailable' in data:
                self.teacher_unavailability[teacher] = data['unavailable']

            for subject in classes:
                if subject in self.subject_to_teachers:
                    self.subject_to_teachers[subject].append(teacher)
        
        # 2. Generate Schedulable Units (The Domain of the Solver Variables)
        # A unit is a unique tuple: (Subject_Name, Teacher_Name, Is_Lab)
        self.schedulable_units = [] 
        self.unit_to_idx = {}
        
        for subject in self.raw_lectures + self.raw_labs:
            teachers = self.subject_to_teachers[subject]
            if not teachers:
                print(f"Warning: No teacher found for {subject}. Assigning 'Staff'.")
                teachers = ["Staff"]
            else:
                print(f"Subject '{subject}' has teachers: {teachers}")
            
            is_lab = subject in self.raw_labs
            
            for teacher in teachers:
                unit = (subject, teacher, is_lab)
                self.schedulable_units.append(unit)
        
        self.num_units = len(self.schedulable_units)
        
        # Helper Maps for Constraints
        self.idx_to_unit = {i: u for i, u in enumerate(self.schedulable_units)}
        
        # Map: Subject Name -> List of Unit Indices (e.g., "DSA" -> [0, 1] where 0 is DSA_SC, 1 is DSA_NewT)
        self.subject_indices_map = {s: [] for s in self.raw_lectures + self.raw_labs}
        
        # Map: Teacher Name -> List of Unit Indices
        self.teacher_indices_map = {}
        
        # --- SMART ROOM ASSIGNMENT LOGIC ---
        self.subject_to_room = {}
        
        # 1. Invert mapping to find all candidate rooms per subject
        subj_to_candidate_rooms = {}
        for room, classes in self.room_assignments.items():
            for c in classes:
                if c not in subj_to_candidate_rooms:
                    subj_to_candidate_rooms[c] = []
                subj_to_candidate_rooms[c].append(room)

        # 2. Track Load on each room to avoid bottlenecks
        # Load = number of slots occupied by all divisions
        room_loads = {r: 0 for r in self.room_assignments.keys()}
        num_divs = len(self.divisions)

        # 3. Assign rooms prioritizing the one with the lowest current load
        print("\n--- Assigning Rooms (Load Balancing) ---")
        for subj in self.raw_lectures + self.raw_labs:
            candidates = subj_to_candidate_rooms.get(subj, [])
            if not candidates:
                print(f"Warning: No room found for {subj}")
                continue

            # Calculate load impact: Freq * Duration * Num_Divisions
            freq = self.frequencies.get(subj, 0)
            duration = 2 if subj in self.raw_labs else 1
            load_impact = freq * duration * num_divs

            # Pick the room with the MINIMUM current load
            best_room = min(candidates, key=lambda r: room_loads[r])
            
            self.subject_to_room[subj] = best_room
            room_loads[best_room] += load_impact
            print(f"Assigned {subj} -> {best_room} (New Load: {room_loads[best_room]})")
        
        # Check for potential overflows (Capacity approx 35 slots/week)
        # 5 days * (End - Start - Lunch)
        slots_per_week = (self.total_slots - (1 if self.lunch_slot_idx is not None else 0)) * len(self.days)
        print(f"\nRoom Capacity Check (Max ~{slots_per_week} slots):")
        for r, load in room_loads.items():
            status = "OK" if load <= slots_per_week else "OVERLOADED!"
            print(f"  {r}: {load} slots [{status}]")

        # Populate Maps
        for idx, (subj, teacher, _) in self.idx_to_unit.items():
            self.subject_indices_map[subj].append(idx)
            
            if teacher not in self.teacher_indices_map:
                self.teacher_indices_map[teacher] = []
            self.teacher_indices_map[teacher].append(idx)

    def build_model(self):
        print("\nBuilding mathematical model with load balancing...")
        
        # 1. Variables: x[div][day][slot] = unit_index
        for d in self.divisions:
            self.variables[d] = {}
            for day in self.days:
                self.variables[d][day] = {}
                for s in self.slots:
                    self.variables[d][day][s] = self.model.NewIntVar(-1, self.num_units - 1, f'x_{d}_{day}_{s}')

        # 2. Lunch Constraint
        if self.lunch_slot_idx is not None:
            for d in self.divisions:
                for day in self.days:
                    self.model.Add(self.variables[d][day][self.lunch_slot_idx] == -1)

        # 3. Division Constraints (Frequency & Load Balancing)
        for d in self.divisions:
            self._add_division_constraints(d)

        # 4. Resource Constraints (Teacher Conflicts & Room Conflicts)
        self._add_resource_constraints()

        # 5. Objective
        self._add_objective_minimize_gaps()

    def _add_division_constraints(self, div):
        # We iterate over SUBJECTS, not units.
        # We need to ensure the SUM of all units for a subject equals the frequency.
        
        # --- LECTURES ---
        for subject in self.raw_lectures:
            if subject not in self.frequencies:
                print(f"Warning: Frequency not defined for {subject}, skipping.")
                continue
            
            freq_required = self.frequencies[subject]
            unit_indices = self.subject_indices_map[subject] # Indices for [(DSA, SC), (DSA, T2)]
            
            # Tracking variables for load balancing
            teacher_usage_vars = {idx: [] for idx in unit_indices}
            all_subject_activations = []

            for day in self.days:
                daily_activations = []
                for s in self.slots:
                    var = self.variables[div][day][s]
                    
                    # Create booleans for each teacher variant of this subject
                    for u_idx in unit_indices:
                        b = self.model.NewBoolVar(f'{div}_{subject}_{day}_{s}_{u_idx}')
                        self.model.Add(var == u_idx).OnlyEnforceIf(b)
                        # Note: We don't enforce "Not" here to save constraints; logic holds naturally
                        
                        teacher_usage_vars[u_idx].append(b)
                        daily_activations.append(b)
                
                # Max 1 lecture of this subject per day (regardless of teacher)
                self.model.Add(sum(daily_activations) <= 1)
                all_subject_activations.extend(daily_activations)

            # A. Total Frequency Constraint
            self.model.Add(sum(all_subject_activations) == freq_required)
            
            # B. Load Balancing (Distribute evenly among available teachers)
            self._apply_load_balancing(unit_indices, teacher_usage_vars, freq_required)

        # --- LABS ---
        for subject in self.raw_labs:
            if subject not in self.frequencies:
                print(f"Warning: Frequency not defined for {subject}, skipping.")
                continue
            
            freq_required = self.frequencies[subject]
            unit_indices = self.subject_indices_map[subject]
            
            teacher_usage_vars = {idx: [] for idx in unit_indices}
            all_lab_starts = []

            for day in self.days:
                daily_starts = []
                for s in range(len(self.slots) - 1):
                    if self.lunch_slot_idx is not None and (s == self.lunch_slot_idx or s + 1 == self.lunch_slot_idx):
                        continue

                    # For labs, we must pick a specific teacher for the ENTIRE duration (s and s+1)
                    for u_idx in unit_indices:
                        b_start = self.model.NewBoolVar(f'start_{div}_{subject}_{day}_{s}_{u_idx}')
                        
                        # If b_start is true, both slots must be this specific unit (Subject+Teacher)
                        self.model.Add(self.variables[div][day][s] == u_idx).OnlyEnforceIf(b_start)
                        self.model.Add(self.variables[div][day][s+1] == u_idx).OnlyEnforceIf(b_start)
                        
                        teacher_usage_vars[u_idx].append(b_start)
                        daily_starts.append(b_start)

                self.model.Add(sum(daily_starts) <= 1)
                all_lab_starts.extend(daily_starts)

            # A. Total Frequency
            self.model.Add(sum(all_lab_starts) == freq_required)

            # B. Load Balancing
            self._apply_load_balancing(unit_indices, teacher_usage_vars, freq_required)

    def _apply_load_balancing(self, unit_indices, teacher_usage_vars, total_freq):
        """Ensures lectures are split evenly. e.g. Freq 4, 2 Teachers -> 2 each."""
        num_teachers = len(unit_indices)
        if num_teachers <= 1: return

        min_load = math.floor(total_freq / num_teachers)
        max_load = math.ceil(total_freq / num_teachers)

        for u_idx in unit_indices:
            # teacher_usage_vars[u_idx] is a list of Bools where this teacher taught
            count_var = sum(teacher_usage_vars[u_idx])
            self.model.Add(count_var >= min_load)
            self.model.Add(count_var <= max_load)

    def _add_resource_constraints(self):
        for day in self.days:
            for s in self.slots:
                active_vars = [self.variables[d][day][s] for d in self.divisions]
                
                # 1. Teacher Clashes
                for teacher, u_indices in self.teacher_indices_map.items():
                    if teacher == "Staff": continue # Don't check conflicts for generic Staff
                    
                    teaching_bools = []
                    for div_var in active_vars:
                        # Teacher is busy if div_var is ANY of the units this teacher teaches
                        # (e.g. DSA_SC or DSL_SC or Math_SC)
                        is_teaching = self.model.NewBoolVar(f'busy_{teacher}_{day}_{s}')
                        
                        # Optimization: AddMaxEquality logic
                        match_bools = []
                        for u_idx in u_indices:
                            b = self.model.NewBoolVar(f'match_{u_idx}')
                            self.model.Add(div_var == u_idx).OnlyEnforceIf(b)
                            self.model.Add(div_var != u_idx).OnlyEnforceIf(b.Not())
                            match_bools.append(b)
                        
                        self.model.AddMaxEquality(is_teaching, match_bools)
                        teaching_bools.append(is_teaching)
                    
                    self.model.Add(sum(teaching_bools) <= 1)

                # 2. Room Clashes
                # We need to map Room -> List of Unit Indices that use this room
                # We built self.subject_to_room, now let's use it
                room_map = {} # Room -> [unit_indices]
                for idx, (subj, _, _) in self.idx_to_unit.items():
                    room = self.subject_to_room.get(subj)
                    if room:
                        if room not in room_map: room_map[room] = []
                        room_map[room].append(idx)
                
                for room, u_indices in room_map.items():
                    occupied_bools = []
                    for div_var in active_vars:
                        is_occ = self.model.NewBoolVar(f'occ_{room}_{day}_{s}')
                        match_bools = []
                        for u_idx in u_indices:
                            b = self.model.NewBoolVar(f'match_r_{u_idx}')
                            self.model.Add(div_var == u_idx).OnlyEnforceIf(b)
                            self.model.Add(div_var != u_idx).OnlyEnforceIf(b.Not())
                            match_bools.append(b)
                        
                        self.model.AddMaxEquality(is_occ, match_bools)
                        occupied_bools.append(is_occ)
                    
                    self.model.Add(sum(occupied_bools) <= 1)

        # 3. Unavailability
        for teacher, unavail_list in self.teacher_unavailability.items():
            u_indices = self.teacher_indices_map.get(teacher, [])
            if not u_indices: continue
            
            for unavail_str in unavail_list:
                try:
                    u_day, u_slot = unavail_str.split('_')
                    u_slot = int(u_slot)
                    if u_day in self.days and u_slot in self.slots:
                        for d in self.divisions:
                            # Cannot assign ANY unit belonging to this teacher
                            for u_idx in u_indices:
                                self.model.Add(self.variables[d][u_day][u_slot] != u_idx)
                except ValueError:
                    pass

    def _add_objective_minimize_gaps(self):
        penalties = []
        for d in self.divisions:
            for day in self.days:
                for s in range(len(self.slots) - 1):
                    if self.lunch_slot_idx is not None and (s == self.lunch_slot_idx - 1 or s == self.lunch_slot_idx):
                        continue 

                    current = self.variables[d][day][s]
                    next_slot = self.variables[d][day][s+1]

                    gap_start = self.model.NewBoolVar(f'gap_start_{d}_{day}_{s}')
                    self.model.Add(current != -1).OnlyEnforceIf(gap_start)
                    self.model.Add(next_slot == -1).OnlyEnforceIf(gap_start)
                    
                    gap_end = self.model.NewBoolVar(f'gap_end_{d}_{day}_{s}')
                    self.model.Add(current == -1).OnlyEnforceIf(gap_end)
                    self.model.Add(next_slot != -1).OnlyEnforceIf(gap_end)

                    penalties.append(gap_start)
                    penalties.append(gap_end)
        
        self.model.Minimize(sum(penalties))

    def solve(self, time_limit_seconds=180, workers=8):
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.num_search_workers = workers
        self.solver.parameters.log_search_progress = True
        
        print(f"Starting solver ({workers} workers, {time_limit_seconds}s limit)...")
        printer = TimetableSolutionPrinter(limit=5)
        status = self.solver.Solve(self.model, printer)
        return status

    def export_solution(self, filename="timetable_full.json"):
        if self.solver.StatusName() not in ["OPTIMAL", "FEASIBLE"]:
            print("No solution to export.")
            return

        timetable_json = {}
        for d in self.divisions:
            table_data = []
            json_data = {}
            time_headers = [f'{self.start_hour+i}:00' for i in self.slots]
            
            print(f'\n=============== Timetable for {d} ===============')
            
            for day in self.days:
                formatted_row, json_row = [], []
                for s in self.slots:
                    if s == self.lunch_slot_idx:
                        json_cell = {"class": "LUNCH", "teacher": "", "room": ""}
                        formatted_cell_str = "LUNCH"
                    else:
                        val = self.solver.Value(self.variables[d][day][s])
                        
                        if val == -1:
                            json_cell = {"class": "Free", "teacher": "", "room": ""}
                            formatted_cell_str = "-"
                        else:
                            # Retrieve details from Unit Index
                            subject, teacher, is_lab = self.idx_to_unit[val]
                            room = self.subject_to_room.get(subject, "")
                            
                            is_continuation = False
                            if s > 0:
                                prev_val = self.solver.Value(self.variables[d][day][s-1])
                                if prev_val == val and is_lab:
                                    is_continuation = True

                            json_cell = {"class": subject, "teacher": teacher, "room": room}
                            
                            if is_continuation:
                                formatted_cell_str = "conti.."
                            else:
                                formatted_cell_str = f"{subject}\n({teacher})"
                    
                    json_row.append(json_cell)
                    formatted_row.append(formatted_cell_str)
                
                table_data.append([day] + formatted_row)
                json_data[day] = json_row
            
            df = pd.DataFrame(table_data, columns=['Day'] + time_headers)
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
            timetable_json[d] = json_data
            
        with open(filename, "w") as f:
            json.dump(timetable_json, f, indent=4)
        print(f"\nSaved to {filename}")

if __name__ == "__main__":
    scheduler = TimetableScheduler()
    try:
        scheduler.build_model()
        status = scheduler.solve()
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            scheduler.export_solution()
        else:
            print("No solution found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
