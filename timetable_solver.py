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
        self.solve_status = None  # Store the solve status
        
        # Variables
        self.var_lectures = {} # x_div[div][day][slot]
        self.var_labs = {}     # x_batch[batch][day][slot]
        
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
        self.raw_lectures = self.config['lectures'] 
        self.raw_labs = self.config['labs']
        self.days = self.config['days']
        
        self.start_hour = self.config['start_hour']
        self.end_hour = self.config['end_hour']
        self.lunch_start = self.config.get('lunch_start_hour')
        self.batches_per_div = self.config.get('batches_per_division', 4)
        
        if self.start_hour >= self.end_hour:
            raise ValueError("Start hour must be before end hour")

        self.total_slots = self.end_hour - self.start_hour
        self.slots = list(range(self.total_slots))
        
        self.lunch_slot_idx = None
        if self.lunch_start and self.start_hour <= self.lunch_start < self.end_hour:
            self.lunch_slot_idx = self.lunch_start - self.start_hour
            print(f"Lunch configured at slot index: {self.lunch_slot_idx} ({self.lunch_start}:00)")

        # Generate Batches
        # Map: DivA -> [A1, A2, A3, A4]
        self.div_to_batches = {}
        self.all_batches = []
        for div in self.divisions:
            # Assuming div name ends with a letter like DivA, DivB... 
            # We can just append 1,2,3,4. 
            # Or simpler: DivA -> DivA_B1, DivA_B2...
            # Let's use the user's notation: A1, A2... if Div is DivA
            base_name = div.replace("Div", "") # DivA -> A
            batches = [f"{base_name}{i+1}" for i in range(self.batches_per_div)]
            self.div_to_batches[div] = batches
            self.all_batches.extend(batches)
        
        print(f"Generated Batches: {self.div_to_batches}")

    def _prepare_solver_data(self):
        print("\n--- Preparing Data ---")
        
        # 1. Invert Teacher Map
        self.subject_to_teachers = {s: [] for s in self.raw_lectures + self.raw_labs}
        self.teacher_unavailability = {}

        for teacher, data in self.teacher_assignments.items():
            classes = data if isinstance(data, list) else data.get('classes', [])
            if isinstance(data, dict) and 'unavailable' in data:
                self.teacher_unavailability[teacher] = data['unavailable']

            for subject in classes:
                if subject in self.subject_to_teachers:
                    self.subject_to_teachers[subject].append(teacher)
        
        # 2. Generate Schedulable Units
        # We separate Lecture Units and Lab Units
        self.lecture_units = []
        self.lab_units = []
        
        # Helper to create units
        def create_units(subjects, is_lab_flag, target_list):
            for subject in subjects:
                teachers = self.subject_to_teachers[subject]
                if not teachers:
                    print(f"Warning: No teacher found for {subject}. Assigning 'Staff'.")
                    teachers = ["Staff"]
                
                for teacher in teachers:
                    unit = (subject, teacher, is_lab_flag)
                    target_list.append(unit)

        create_units(self.raw_lectures, False, self.lecture_units)
        create_units(self.raw_labs, True, self.lab_units)
        
        self.num_lecture_units = len(self.lecture_units)
        self.num_lab_units = len(self.lab_units)
        
        # Lookups
        self.idx_to_lecture_unit = {i: u for i, u in enumerate(self.lecture_units)}
        self.idx_to_lab_unit = {i: u for i, u in enumerate(self.lab_units)}
        
        # Map: Subject -> List of Unit Indices
        self.subj_to_lecture_indices = {s: [] for s in self.raw_lectures}
        for idx, (subj, _, _) in self.idx_to_lecture_unit.items():
            self.subj_to_lecture_indices[subj].append(idx)
            
        self.subj_to_lab_indices = {s: [] for s in self.raw_labs}
        for idx, (subj, _, _) in self.idx_to_lab_unit.items():
            self.subj_to_lab_indices[subj].append(idx)

        # Map: Teacher -> List of (Type, Index)
        # Type: 'lec' or 'lab'
        self.teacher_to_units = {} 
        
        for idx, (_, teacher, _) in self.idx_to_lecture_unit.items():
            if teacher not in self.teacher_to_units: self.teacher_to_units[teacher] = []
            self.teacher_to_units[teacher].append(('lec', idx))
            
        for idx, (_, teacher, _) in self.idx_to_lab_unit.items():
            if teacher not in self.teacher_to_units: self.teacher_to_units[teacher] = []
            self.teacher_to_units[teacher].append(('lab', idx))

        # --- ROOM ASSIGNMENT ---
        # We keep the static assignment for now, but we need to handle it carefully.
        # Ideally, rooms should be dynamic for labs to allow parallel sessions.
        # BUT, the user asked for "assign some batches and labs parallely in same timing slots, based on availability".
        # This implies dynamic room choice OR multiple rooms per subject.
        # Current rooms.json maps Room -> [Subjects].
        # Let's Invert: Subject -> [List of Candidate Rooms]
        
        self.subj_to_rooms = {}
        for room, classes in self.room_assignments.items():
            for c in classes:
                if c not in self.subj_to_rooms: self.subj_to_rooms[c] = []
                self.subj_to_rooms[c].append(room)
        
        # For Lectures: Pick 1 BEST room per subject (Load Balancing) - Same as before
        # For Labs: We need to allow ANY valid room to be picked dynamically by the solver?
        # OR we pre-assign. Pre-assigning is risky for parallel batches.
        # Let's make Room a Variable for Labs? Or just iterate over rooms?
        # Complexity increase: If we make room a variable, it explodes.
        # Alternative: Pre-assign multiple rooms to a lab subject if available.
        # E.g. DEVL has 9001, 9002... 
        # We can create Lab Units as (Subject, Teacher, Room).
        # This expands the domain but solves the collision issue.
        
        print("\n--- Expanding Lab Units with Rooms ---")
        new_lab_units = []
        for (subj, teacher, _) in self.lab_units:
            candidate_rooms = self.subj_to_rooms.get(subj, [])
            if not candidate_rooms:
                print(f"Warning: No room for lab {subj}")
                candidate_rooms = ["AnyRoom"]
            
            for room in candidate_rooms:
                # New Unit: (Subject, Teacher, Room)
                # We mark is_lab=True
                new_lab_units.append((subj, teacher, room))
        
        self.lab_units = new_lab_units
        self.num_lab_units = len(self.lab_units)
        self.idx_to_lab_unit = {i: u for i, u in enumerate(self.lab_units)}
        
        # Re-map subject indices for labs4
        print("Re-mapping Lab Unit Indices...")
        self.subj_to_lab_indices = {s: [] for s in self.raw_labs}
        for idx, (subj, _, _) in self.idx_to_lab_unit.items():
            self.subj_to_lab_indices[subj].append(idx)
            
        # Re-map teacher indices for labs
        print("Re-mapping Teacher to Units...")
        self.teacher_to_units = {} # Reset and rebuild
        for idx, (_, teacher, _) in self.idx_to_lecture_unit.items():
            if teacher not in self.teacher_to_units: self.teacher_to_units[teacher] = []
            self.teacher_to_units[teacher].append(('lec', idx))
            
        for idx, (_, teacher, _) in self.idx_to_lab_unit.items():
            if teacher not in self.teacher_to_units: self.teacher_to_units[teacher] = []
            self.teacher_to_units[teacher].append(('lab', idx))

        # Room -> List of (Type, Index)
        self.room_to_units = {}
        # For lectures, we still need to assign a room. Let's do the static assignment for lectures only.
        self.lecture_subj_to_room = {}
        room_loads = {r: 0 for r in self.room_assignments.keys()}
        
        for subj in self.raw_lectures:
            candidates = self.subj_to_rooms.get(subj, [])
            if candidates:
                best_room = min(candidates, key=lambda r: room_loads.get(r, 0))
                self.lecture_subj_to_room[subj] = best_room
                room_loads[best_room] = room_loads.get(best_room, 0) + self.frequencies.get(subj, 0) * len(self.divisions)
                
                # Register usage
                if best_room not in self.room_to_units: self.room_to_units[best_room] = []
                # Add all units of this subject
                for idx in self.subj_to_lecture_indices[subj]:
                    self.room_to_units[best_room].append(('lec', idx))
            else:
                self.lecture_subj_to_room[subj] = "Unassigned"

        # For labs, the room is part of the unit
        print("Mapping Rooms to Lab Units...")
        for idx, (_, _, room) in self.idx_to_lab_unit.items():
            if room not in self.room_to_units: self.room_to_units[room] = []
            self.room_to_units[room].append(('lab', idx))

    def build_model(self):
        print("\nBuilding Batch-Enabled Model...")
        
        # 1. Variables
        # A. Lectures: x_div[div][day][slot] -> Lecture Unit Index
        for div in self.divisions:
            self.var_lectures[div] = {}
            for day in self.days:
                self.var_lectures[div][day] = {}
                for s in self.slots:
                    self.var_lectures[div][day][s] = self.model.NewIntVar(-1, self.num_lecture_units - 1, f'Lec_{div}_{day}_{s}')

        # B. Labs: x_batch[batch][day][slot] -> Lab Unit Index
        print("Creating Lab Variables...")
        for batch in self.all_batches:
            self.var_labs[batch] = {}
            for day in self.days:
                self.var_labs[batch][day] = {}
                for s in self.slots:
                    self.var_labs[batch][day][s] = self.model.NewIntVar(-1, self.num_lab_units - 1, f'Lab_{batch}_{day}_{s}')

        # 2. Lunch Constraint
        print("Adding Lunch Constraints...")
        if self.lunch_slot_idx is not None:
            for div in self.divisions:
                for day in self.days:
                    self.model.Add(self.var_lectures[div][day][self.lunch_slot_idx] == -1)
            for batch in self.all_batches:
                for day in self.days:
                    self.model.Add(self.var_labs[batch][day][self.lunch_slot_idx] == -1)

        # 3. Hierarchy Constraint: If Division has Lecture, Batches cannot have Lab
        print("Adding Hierarchy Constraints...")
        for div in self.divisions:
            batches = self.div_to_batches[div]
            for day in self.days:
                for s in self.slots:
                    lec_var = self.var_lectures[div][day][s]
                    is_lecture = self.model.NewBoolVar(f'is_lec_{div}_{day}_{s}')
                    self.model.Add(lec_var != -1).OnlyEnforceIf(is_lecture)
                    self.model.Add(lec_var == -1).OnlyEnforceIf(is_lecture.Not())
                    
                    for b in batches:
                        lab_var = self.var_labs[b][day][s]
                        # If Lecture is ON, Lab must be OFF (-1)
                        self.model.Add(lab_var == -1).OnlyEnforceIf(is_lecture)

        # 4. Division Constraints (Lectures)
        print("Adding Lecture Constraints...")
        for div in self.divisions:
            self._add_lecture_constraints(div)

        # 5. Batch Constraints (Labs)
        print("Adding Lab Constraints...")
        for batch in self.all_batches:
            self._add_lab_constraints(batch)

        # 6. Resource Constraints (Teachers & Rooms)
        print("Adding Resource Constraints...")
        self._add_resource_constraints()

        # 7. Synchronization Constraints
        print("Adding Lab Synchronization Constraints...")
        self._add_lab_synchronization()

        # 8. Objective
        print("Adding Objective...")
        self._add_objective()

    def _add_lecture_constraints(self, div):
        for subject in self.raw_lectures:
            if subject not in self.frequencies: continue
            freq = self.frequencies[subject]
            unit_indices = self.subj_to_lecture_indices[subject]
            
            # Group unit indices by teacher to ensure one teacher per division-subject
            teacher_to_unit_indices = {}
            for u_idx in unit_indices:
                _, teacher, _ = self.idx_to_lecture_unit[u_idx]
                if teacher not in teacher_to_unit_indices:
                    teacher_to_unit_indices[teacher] = []
                teacher_to_unit_indices[teacher].append(u_idx)
            
            # Create boolean variable for each teacher: is this teacher selected for this div-subject?
            teacher_selected = {}
            for teacher in teacher_to_unit_indices.keys():
                teacher_selected[teacher] = self.model.NewBoolVar(f'teacher_sel_{div}_{subject}_{teacher}')
            
            # Constraint: Exactly one teacher must be selected for this division-subject pair
            if len(teacher_selected) > 1:
                self.model.Add(sum(teacher_selected.values()) == 1)
            elif len(teacher_selected) == 1:
                # Only one teacher available, must be selected
                self.model.Add(list(teacher_selected.values())[0] == 1)
            
            all_activations = []
            for day in self.days:
                daily_acts = []
                for s in self.slots:
                    var = self.var_lectures[div][day][s]
                    for u_idx in unit_indices:
                        _, teacher, _ = self.idx_to_lecture_unit[u_idx]
                        b = self.model.NewBoolVar(f'lec_act_{div}_{subject}_{day}_{s}_{u_idx}')
                        # Bidirectional: b <=> (var == u_idx)
                        self.model.Add(var == u_idx).OnlyEnforceIf(b)
                        self.model.Add(var != u_idx).OnlyEnforceIf(b.Not())
                        
                        # Only allow this unit if its teacher is selected
                        if teacher in teacher_selected:
                            self.model.Add(b == 0).OnlyEnforceIf(teacher_selected[teacher].Not())
                        
                        daily_acts.append(b)
                
                self.model.Add(sum(daily_acts) <= 1)
                all_activations.extend(daily_acts)
            
            self.model.Add(sum(all_activations) == freq)



    def _add_lab_constraints(self, batch):
        for subject in self.raw_labs:
            if subject not in self.frequencies: continue
            freq = self.frequencies[subject]
            num_sessions = freq
            
            unit_indices = self.subj_to_lab_indices[subject]
            
            # Group unit indices by teacher to ensure one teacher per batch-subject
            teacher_to_unit_indices = {}
            for u_idx in unit_indices:
                _, teacher, _ = self.idx_to_lab_unit[u_idx]
                if teacher not in teacher_to_unit_indices:
                    teacher_to_unit_indices[teacher] = []
                teacher_to_unit_indices[teacher].append(u_idx)
            
            # Create boolean variable for each teacher: is this teacher selected for this batch-subject?
            teacher_selected = {}
            for teacher in teacher_to_unit_indices.keys():
                teacher_selected[teacher] = self.model.NewBoolVar(f'lab_teacher_sel_{batch}_{subject}_{teacher}')
            
            # Constraint: Exactly one teacher must be selected for this batch-subject pair
            if len(teacher_selected) > 1:
                self.model.Add(sum(teacher_selected.values()) == 1)
            elif len(teacher_selected) == 1:
                # Only one teacher available, must be selected
                self.model.Add(list(teacher_selected.values())[0] == 1)
            
            all_starts = []
            
            # Map: (day, slot, u_idx) -> b_start (that STARTS at this slot)
            starts_at = {} 

            for day in self.days:
                daily_starts = []
                for s in range(len(self.slots) - 1):
                    if self.lunch_slot_idx is not None and (s == self.lunch_slot_idx or s + 1 == self.lunch_slot_idx):
                        continue
                    
                    for u_idx in unit_indices:
                        _, teacher, _ = self.idx_to_lab_unit[u_idx]
                        b_start = self.model.NewBoolVar(f'lab_start_{batch}_{subject}_{day}_{s}_{u_idx}')
                        
                        # Forward: Start => Occupy s and s+1
                        self.model.Add(self.var_labs[batch][day][s] == u_idx).OnlyEnforceIf(b_start)
                        self.model.Add(self.var_labs[batch][day][s+1] == u_idx).OnlyEnforceIf(b_start)
                        
                        # Only allow this unit if its teacher is selected
                        if teacher in teacher_selected:
                            self.model.Add(b_start == 0).OnlyEnforceIf(teacher_selected[teacher].Not())
                        
                        daily_starts.append(b_start)
                        starts_at[(day, s, u_idx)] = b_start
                
                self.model.Add(sum(daily_starts) <= 1)
                all_starts.extend(daily_starts)
            
            self.model.Add(sum(all_starts) == num_sessions)

            # Reverse: Occupy => Start at s OR Start at s-1
            for day in self.days:
                for s in self.slots:
                    if s == self.lunch_slot_idx: continue
                    
                    for u_idx in unit_indices:
                        _, teacher, _ = self.idx_to_lab_unit[u_idx]
                        
                        # Only consider units from the selected teacher
                        if teacher not in teacher_selected:
                            continue
                        
                        # If var == u_idx, then...
                        # It could be a start at s: starts_at.get((day, s, u_idx))
                        # OR a start at s-1: starts_at.get((day, s-1, u_idx))
                        
                        potential_causes = []
                        if (day, s, u_idx) in starts_at:
                            potential_causes.append(starts_at[(day, s, u_idx)])
                        if (day, s-1, u_idx) in starts_at:
                            potential_causes.append(starts_at[(day, s-1, u_idx)])
                        
                        if potential_causes:
                            # var == u_idx  ==>  sum(causes) >= 1
                            # equivalent: var != u_idx OR sum(causes) >= 1
                            # easier: OnlyEnforceIf(var == u_idx) -> sum(causes) >= 1
                            
                            is_assigned = self.model.NewBoolVar(f'is_assigned_{batch}_{day}_{s}_{u_idx}')
                            self.model.Add(self.var_labs[batch][day][s] == u_idx).OnlyEnforceIf(is_assigned)
                            self.model.Add(self.var_labs[batch][day][s] != u_idx).OnlyEnforceIf(is_assigned.Not())
                            
                            # If assigned, then one of the causes must be true
                            self.model.Add(sum(potential_causes) >= 1).OnlyEnforceIf(is_assigned)
                            
                            # Also ensure teacher is selected if this unit is assigned
                            self.model.Add(is_assigned == 0).OnlyEnforceIf(teacher_selected[teacher].Not())
                        else:
                            # If no potential starts cover this slot (e.g. lunch edge cases?), then it CANNOT be assigned
                            self.model.Add(self.var_labs[batch][day][s] != u_idx)

    def _add_resource_constraints(self):
        # 1. Teachers
        for teacher, units in self.teacher_to_units.items():
            if teacher == "Staff": continue
            
            for day in self.days:
                for s in self.slots:
                    # Gather all booleans where this teacher is active
                    active_bools = []
                    
                    # A. Lectures
                    for div in self.divisions:
                        var = self.var_lectures[div][day][s]
                        # Is var assigned to any unit taught by this teacher?
                        # Optimization: Pre-calculate relevant indices
                        relevant_lec_indices = [idx for (t, idx) in units if t == 'lec']
                        if relevant_lec_indices:
                            # Create a bool that is true if var is in relevant_indices
                            # Since domain is small, we can iterate
                            for idx in relevant_lec_indices:
                                b = self.model.NewBoolVar(f't_busy_lec_{teacher}_{div}_{day}_{s}_{idx}')
                                self.model.Add(var == idx).OnlyEnforceIf(b)
                                self.model.Add(var != idx).OnlyEnforceIf(b.Not())
                                active_bools.append(b)

                    # B. Labs
                    for batch in self.all_batches:
                        var = self.var_labs[batch][day][s]
                        relevant_lab_indices = [idx for (t, idx) in units if t == 'lab']
                        if relevant_lab_indices:
                            for idx in relevant_lab_indices:
                                b = self.model.NewBoolVar(f't_busy_lab_{teacher}_{batch}_{day}_{s}_{idx}')
                                self.model.Add(var == idx).OnlyEnforceIf(b)
                                self.model.Add(var != idx).OnlyEnforceIf(b.Not())
                                active_bools.append(b)
                    
                    # Constraint: At most 1 active assignment
                    if active_bools:
                        self.model.Add(sum(active_bools) <= 1)

        # 2. Rooms
        for room, units in self.room_to_units.items():
            for day in self.days:
                for s in self.slots:
                    occupied_bools = []
                    
                    # A. Lectures
                    relevant_lec_indices = [idx for (t, idx) in units if t == 'lec']
                    if relevant_lec_indices:
                        for div in self.divisions:
                            var = self.var_lectures[div][day][s]
                            for idx in relevant_lec_indices:
                                b = self.model.NewBoolVar(f'r_occ_lec_{room}_{div}_{day}_{s}_{idx}')
                                self.model.Add(var == idx).OnlyEnforceIf(b)
                                self.model.Add(var != idx).OnlyEnforceIf(b.Not())
                                occupied_bools.append(b)
                    
                    # B. Labs
                    relevant_lab_indices = [idx for (t, idx) in units if t == 'lab']
                    if relevant_lab_indices:
                        for batch in self.all_batches:
                            var = self.var_labs[batch][day][s]
                            for idx in relevant_lab_indices:
                                b = self.model.NewBoolVar(f'r_occ_lab_{room}_{batch}_{day}_{s}_{idx}')
                                self.model.Add(var == idx).OnlyEnforceIf(b)
                                self.model.Add(var != idx).OnlyEnforceIf(b.Not())
                                occupied_bools.append(b)
                    
                    if occupied_bools:
                        self.model.Add(sum(occupied_bools) <= 1)

    def _add_division_lab_teacher_consistency(self):
        """
        Ensure that for a given Division and Lab Subject, all batches in that division
        use the same teacher. This ensures consistency across batches within a division.
        """
        for div in self.divisions:
            batches = self.div_to_batches[div]
            if len(batches) <= 1:
                continue  # No need to enforce consistency for single batch
            
            for subject in self.raw_labs:
                if subject not in self.frequencies:
                    continue
                
                unit_indices = self.subj_to_lab_indices[subject]
                if not unit_indices:
                    continue
                
                # Get all teachers for this subject
                teachers = set()
                for u_idx in unit_indices:
                    _, teacher, _ = self.idx_to_lab_unit[u_idx]
                    teachers.add(teacher)
                
                if len(teachers) <= 1:
                    continue  # Only one teacher available, no need to enforce
                
                # Create division-level teacher selection variable
                div_teacher_selected = {}
                for teacher in teachers:
                    div_teacher_selected[teacher] = self.model.NewBoolVar(f'div_lab_teacher_{div}_{subject}_{teacher}')
                
                # Exactly one teacher selected at division level
                self.model.Add(sum(div_teacher_selected.values()) == 1)
                
                # For each batch, ensure it can only use units from the division-selected teacher
                for batch in batches:
                    for u_idx in unit_indices:
                        _, unit_teacher, _ = self.idx_to_lab_unit[u_idx]
                        
                        # If division selected a teacher other than unit_teacher, 
                        # then this batch cannot use this unit
                        for teacher in teachers:
                            if teacher != unit_teacher:
                                # If div selected teacher, batch cannot use units from other teachers
                                for day in self.days:
                                    for s in self.slots:
                                        self.model.Add(
                                            self.var_labs[batch][day][s] != u_idx
                                        ).OnlyEnforceIf(div_teacher_selected[teacher])

    def _add_lab_synchronization(self):
        """
        Ensure that for a given Division and Day, if multiple batches have labs,
        they all start at the same time slot.
        """
        print("Adding Lab Synchronization Constraints (Fixed)...")
        for div in self.divisions:
            batches = self.div_to_batches[div]
            
            for day in self.days:
                # Create boolean variables for "Division starts lab at slot s"
                div_starts = []
                for s in self.slots:
                    div_starts.append(self.model.NewBoolVar(f'div_lab_start_{div}_{day}_{s}'))
                
                # Collect all batch start vars for each slot
                batch_starts_by_slot = {s: [] for s in self.slots}

                for batch in batches:
                    # Create is_active vars
                    is_active = [] 
                    for s in self.slots:
                        b_act = self.model.NewBoolVar(f'b_act_{batch}_{day}_{s}')
                        var = self.var_labs[batch][day][s]
                        self.model.Add(var != -1).OnlyEnforceIf(b_act)
                        self.model.Add(var == -1).OnlyEnforceIf(b_act.Not())
                        is_active.append(b_act)
                    
                    for s in self.slots:
                        if s == self.lunch_slot_idx: continue
                        
                        b_start = self.model.NewBoolVar(f'b_start_{batch}_{day}_{s}')
                        
                        # Bidirectional binding: b_start <=> (Active[s] AND !Active[s-1])
                        
                        # 1. If Active[s] is False -> b_start is False
                        self.model.Add(b_start == 0).OnlyEnforceIf(is_active[s].Not())
                        
                        # 2. If Active[s-1] is True -> b_start is False (Continuation)
                        if s > 0:
                            self.model.Add(b_start == 0).OnlyEnforceIf(is_active[s-1])
                        
                        # 3. If (Active[s] AND !Active[s-1]) -> b_start is True
                        conditions = [is_active[s]]
                        if s > 0:
                            conditions.append(is_active[s-1].Not())
                        self.model.Add(b_start == 1).OnlyEnforceIf(conditions)
                        
                        batch_starts_by_slot[s].append(b_start)

                # Link Batch Starts to Division Starts
                # div_starts[s] <=> max(batch_starts_by_slot[s])
                for s in self.slots:
                    if batch_starts_by_slot[s]:
                        self.model.AddMaxEquality(div_starts[s], batch_starts_by_slot[s])
                    else:
                        self.model.Add(div_starts[s] == 0)

                # Constraint: Prevent Staggered Starts
                # We cannot have a start at s and a start at s+1
                # This prevents 9-11 and 10-12 overlaps
                for s in range(len(self.slots) - 1):
                    # If div starts at s, it cannot start at s+1
                    self.model.Add(div_starts[s] + div_starts[s+1] <= 1)

    def _add_objective(self):
        # Minimize gaps for students (Divisions)
        # We only check gaps in the merged schedule (Lecture + Lab)
        # But wait, a student in Batch A1 attends DivA lectures AND A1 labs.
        # So we should check gaps for each Batch.
        
        penalties = []
        for batch in self.all_batches:
            # Find parent div
            parent_div = next(d for d, bs in self.div_to_batches.items() if batch in bs)
            
            for day in self.days:
                for s in range(len(self.slots) - 1):
                    if self.lunch_slot_idx is not None and (s == self.lunch_slot_idx - 1 or s == self.lunch_slot_idx):
                        continue
                    
                    # Is slot s occupied?
                    # Occupied if (Div Lecture at s) OR (Batch Lab at s)
                    # Since they are mutually exclusive, we can sum bools?
                    # Or just check if either var != -1
                    
                    def is_occupied(d, b, slot, name_suffix):
                        # Returns a BoolVar that is true if occupied
                        is_occ = self.model.NewBoolVar(f'is_occ_{name_suffix}')
                        
                        # Logic: is_occ <=> (lec != -1 OR lab != -1)
                        # Since they are exclusive, sum(is_active) == 1 or 0
                        
                        lec_var = self.var_lectures[d][day][slot]
                        lab_var = self.var_labs[b][day][slot]
                        
                        lec_active = self.model.NewBoolVar(f'lec_act_{name_suffix}')
                        self.model.Add(lec_var != -1).OnlyEnforceIf(lec_active)
                        self.model.Add(lec_var == -1).OnlyEnforceIf(lec_active.Not())
                        
                        lab_active = self.model.NewBoolVar(f'lab_act_{name_suffix}')
                        self.model.Add(lab_var != -1).OnlyEnforceIf(lab_active)
                        self.model.Add(lab_var == -1).OnlyEnforceIf(lab_active.Not())
                        
                        self.model.Add(is_occ == 1).OnlyEnforceIf(lec_active)
                        self.model.Add(is_occ == 1).OnlyEnforceIf(lab_active)
                        self.model.Add(is_occ == 0).OnlyEnforceIf([lec_active.Not(), lab_active.Not()])
                        
                        return is_occ

                    curr_occ = is_occupied(parent_div, batch, s, f'{batch}_{day}_{s}')
                    next_occ = is_occupied(parent_div, batch, s+1, f'{batch}_{day}_{s+1}')
                    
                    # Gap: Current occupied, Next Empty, Next+1 Occupied? 
                    # Simple gap: Occ, Empty, Occ.
                    # Let's stick to the previous simple logic: Minimize transitions?
                    # Or just minimize isolated empty slots between classes.
                    # Let's just minimize (Occ, Empty) followed by (Occ) later? Too complex.
                    # Let's just minimize "Gap Start" where Occ -> Empty (unless it's end of day)
                    # Actually, let's skip objective for now to ensure feasibility first, or keep it simple.
                    pass 
        
        # self.model.Minimize(sum(penalties))
        pass

    def solve(self):
        print("Solving...")
        self.solver.parameters.max_time_in_seconds = 300
        self.solver.parameters.num_search_workers = 8
        self.solve_status = self.solver.Solve(self.model, TimetableSolutionPrinter(5))
        return self.solve_status

    def export_solution(self, filename="timetable_full.json"):
        if self.solve_status is None or self.solve_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("No solution.")
            return

        full_data = {}
        
        # We want to show the schedule for each Division, but now it's hierarchical.
        # Maybe show per Batch?
        
        for div in self.divisions:
            # print(f"\n=== {div} Schedule ===")
            div_data = {}
            
            # We can print a table where rows are Batches
            # Columns are Slots
            
            headers = ["Batch"] + [f"{self.start_hour+s}:00" for s in self.slots]
            table_rows = []
            
            batches = self.div_to_batches[div]
            
            """
            for day in self.days:
                print(f"\n--- {day} ---")
                day_rows = []
                
                for batch in batches:
                    row = [batch]
                    for s in self.slots:
                        if s == self.lunch_slot_idx:
                            row.append("LUNCH")
                            continue
                        
                        # Check Lecture
                        lec_val = self.solver.Value(self.var_lectures[div][day][s])
                        if lec_val != -1:
                            subj, teacher, _ = self.idx_to_lecture_unit[lec_val]
                            room = self.lecture_subj_to_room.get(subj, "")
                            row.append(f"LEC: {subj}\n({teacher}, {room})")
                        else:
                            # Check Lab
                            lab_val = self.solver.Value(self.var_labs[batch][day][s])
                            if lab_val != -1:
                                subj, teacher, room = self.idx_to_lab_unit[lab_val]
                                row.append(f"LAB: {subj}\n({teacher}, {room})")
                            else:
                                row.append("-")
                    day_rows.append(row)
                
                print(tabulate(day_rows, headers=headers, tablefmt="grid"))
                """
                
                
        # Save JSON
        timetable_json = {}
        for div in self.divisions:
            div_data = {}
            batches = self.div_to_batches[div]
            
            # We structure the JSON by Batches, as that's the most granular unit
            # But we also want to know the Division schedule.
            # Let's do: { "DivA": { "A1": { "Mon": [...] }, "A2": ... } }
            
            batch_data = {}
            for batch in batches:
                batch_schedule = {}
                for day in self.days:
                    day_slots = []
                    for s in self.slots:
                        if s == self.lunch_slot_idx:
                            day_slots.append({"class": "LUNCH", "type": "COMMON"})
                            continue
                        
                        # Check Lecture (Common)
                        lec_val = self.solver.Value(self.var_lectures[div][day][s])
                        if lec_val != -1:
                            subj, teacher, _ = self.idx_to_lecture_unit[lec_val]
                            room = self.lecture_subj_to_room.get(subj, "")
                            day_slots.append({
                                "class": subj,
                                "type": "LECTURE",
                                "teacher": teacher,
                                "room": room
                            })
                        else:
                            # Check Lab (Batch specific)
                            lab_val = self.solver.Value(self.var_labs[batch][day][s])
                            if lab_val != -1:
                                subj, teacher, room = self.idx_to_lab_unit[lab_val]
                                day_slots.append({
                                    "class": subj,
                                    "type": "LAB",
                                    "teacher": teacher,
                                    "room": room
                                })
                            else:
                                day_slots.append({"class": "Free", "type": "FREE"})
                    
                    batch_schedule[day] = day_slots
                batch_data[batch] = batch_schedule
            
            timetable_json[div] = batch_data

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
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
