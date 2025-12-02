from ortools.sat.python import cp_model
import pandas as pd
import json
import time

def generate_timetable(data):
    """
    data = {
        "settings": {...},
        "frequencies": {...},
        "teachers": {...},
        "rooms": {...}
    }
    """

    # -------------------------------
    # LOAD INPUTS FROM FASTAPI JSON
    # -------------------------------
     # SETTINGS
    config = data["settings"]
    days = config["days"]
    start_hour = config["start_hour"]
    end_hour = config["end_hour"]
    lunch_start_hour = config.get("lunch_start_hour")

    # DIVISIONS
    divisions = [d["name"] for d in data["divisions"]]

    # LECTURES + LABS
    subjects = data["subjects"]
    labs_data = data["labs"]

    lectures = [s["name"] for s in subjects]
    labs = [l["name"] for l in labs_data]

    # FREQUENCIES
    class_frequencies = {}
    for s in subjects:
        class_frequencies[s["name"]] = s["frequency"]
    for l in labs_data:
        class_frequencies[l["name"]] = l["frequency"]

    # TEACHER ASSIGNMENTS
    teacher_assignments = {}
    for t in data["teachers"]:
        teacher_assignments[t["name"]] = [sub["name"] for sub in t["subjects"]]

    # ROOM ASSIGNMENTS
    room_assignments = {}
    for r in data["classrooms"]:
        room_assignments[r["name"]] = [sub["name"] for sub in r["subjects"]]

    # BELOW THIS — your original logic continues
    total_hours = end_hour - start_hour

    slots = list(range(total_hours))

    # -------------------------------
    # LUNCH SLOT CALCULATION
    # -------------------------------
    LUNCH_SLOT = None
    if lunch_start_hour is not None and start_hour <= lunch_start_hour < end_hour:
        LUNCH_SLOT = lunch_start_hour - start_hour

    all_classes = lectures + labs
    num_classes = len(all_classes)
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    class_to_teacher = {cls: teacher for teacher, classes in teacher_assignments.items() for cls in classes}
    class_to_room = {cls: room for room, classes in room_assignments.items() for cls in classes}

    all_teachers = list(teacher_assignments.keys())
    all_rooms = list(room_assignments.keys())

    # -------------------------------
    # MODEL
    # -------------------------------
    model = cp_model.CpModel()
    x = {}

    for d in divisions:
        x[d] = {}
        for day in days:
            x[d][day] = {}
            for s in slots:
                x[d][day][s] = model.NewIntVar(-1, num_classes - 1, f"x_{d}_{day}_{s}")

    # Lunch break constraint
    if LUNCH_SLOT is not None:
        for d in divisions:
            for day in days:
                model.Add(x[d][day][LUNCH_SLOT] == -1)

    # -------------------------------
    # LECTURE & LAB CONSTRAINTS
    # -------------------------------
    for d in divisions:

        # LABS (2-hour blocks)
        for lab in labs:
            lab_index = class_to_idx[lab]
            starts_week = []

            for day in days:
                starts_day = []
                for s in range(len(slots) - 1):
                    if LUNCH_SLOT is not None and (s == LUNCH_SLOT or s + 1 == LUNCH_SLOT):
                        continue

                    b = model.NewBoolVar(f"lab_{d}_{lab}_{day}_{s}")
                    starts_day.append(b)

                    model.Add(x[d][day][s] == lab_index).OnlyEnforceIf(b)
                    model.Add(x[d][day][s+1] == lab_index).OnlyEnforceIf(b)

                model.Add(sum(starts_day) <= 1)
                starts_week.extend(starts_day)

            model.Add(sum(starts_week) == class_frequencies[lab])

        # LECTURES
        for lec in lectures:
            lec_index = class_to_idx[lec]
            week_bools = []

            for day in days:
                day_bools = []

                for s in slots:
                    b = model.NewBoolVar(f"lec_{d}_{lec}_{day}_{s}")
                    day_bools.append(b)

                    model.Add(x[d][day][s] == lec_index).OnlyEnforceIf(b)

                model.Add(sum(day_bools) <= 1)
                week_bools.extend(day_bools)

            model.Add(sum(week_bools) == class_frequencies[lec])

    # -------------------------------
    # TEACHER & ROOM CLASHES
    # -------------------------------
    for day in days:
        for s in slots:

            slot_vars_all = [x[d][day][s] for d in divisions]

            # Teacher clashes
            for teacher in all_teachers:
                class_list = [class_to_idx[c] for c in teacher_assignments[teacher]]
                bools = []

                for c_idx in class_list:
                    for v in slot_vars_all:
                        b = model.NewBoolVar("")
                        model.Add(v == c_idx).OnlyEnforceIf(b)
                        model.Add(v != c_idx).OnlyEnforceIf(b.Not())
                        bools.append(b)

                model.Add(sum(bools) <= 1)

            # Room clashes
            for room in all_rooms:
                class_list = [class_to_idx[c] for c in room_assignments[room]]
                bools = []

                for c_idx in class_list:
                    for v in slot_vars_all:
                        b = model.NewBoolVar("")
                        model.Add(v == c_idx).OnlyEnforceIf(b)
                        model.Add(v != c_idx).OnlyEnforceIf(b.Not())
                        bools.append(b)

                model.Add(sum(bools) <= 1)

    # -------------------------------
    # SOLVE
    # -------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"error": "No feasible timetable found"}

    # -------------------------------
    # BUILD OUTPUT JSON
    # -------------------------------
    final_json = {}

    for d in divisions:
        div_json = {}

        for day in days:
            row = []

            for s in slots:
                if LUNCH_SLOT == s:
                    row.append({"class": "LUNCH", "teacher": "", "room": ""})
                    continue

                val = solver.Value(x[d][day][s])

                if val == -1:
                    row.append({"class": "FREE", "teacher": "", "room": ""})
                else:
                    cls = idx_to_class[val]
                    row.append({
                        "class": cls,
                        "teacher": class_to_teacher.get(cls, ""),
                        "room": class_to_room.get(cls, "")
                    })

            div_json[day] = row

        final_json[d] = div_json

    return final_json
