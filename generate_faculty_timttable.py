import json
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    if not os.path.exists('timetable_full.json'):
        print("Error: timetable_full.json not found.")
        return

    timetable = load_json('timetable_full.json')
    config = load_json('config.json')

    days = config['days']
    start_hour = config['start_hour']
    end_hour = config['end_hour']
    total_slots = end_hour - start_hour

    # Structure: Teacher -> Day -> SlotIndex -> Info
    faculty_schedule = {}

    # 1. Iterate through the student timetable to build faculty schedule
    for div, batches in timetable.items():
        for batch, day_sched in batches.items():
            for day, slots in day_sched.items():
                if day not in days: continue

                for slot_idx, slot_data in enumerate(slots):
                    if slot_idx >= total_slots: break # Safety

                    # We are looking for valid assignments with a teacher
                    # Skip FREE, LUNCH (unless we want to mark lunch for teachers? Usually teachers have lunch when free, but let's skip specific "LUNCH" blocks as they are student-centric. Teachers might have duties, but usually it's just a common break. We can mark it if we want.)
                    # The user wants "when is she/he free?". LUNCH is usually free time or specific break.
                    # Let's ignore "LUNCH" type for now, or mark it as "LUNCH".
                    
                    if not isinstance(slot_data, dict):
                        continue
                        
                    teacher = slot_data.get('teacher')
                    if not teacher:
                        continue
                    
                    # We have a teacher for this slot
                    if teacher not in faculty_schedule:
                        faculty_schedule[teacher] = {d: [None]*total_slots for d in days}
                    
                    current_entry = faculty_schedule[teacher][day][slot_idx]
                    
                    # Construct entry data
                    subject = slot_data.get('class', 'Unknown')
                    room = slot_data.get('room', 'Unknown')
                    sType = slot_data.get('type', 'Unknown')
                    
                    if current_entry is None:
                        # New entry
                        faculty_schedule[teacher][day][slot_idx] = {
                            "subject": subject,
                            "room": room,
                            "type": sType,
                            "batches": [batch],
                            "divisions": [div]
                        }
                    else:
                        # Update existing entry (Merge batches)
                        # Check if it's effectively the same class (same room, same subject)
                        if current_entry['room'] == room and current_entry['subject'] == subject:
                            if batch not in current_entry['batches']:
                                current_entry['batches'].append(batch)
                            if div not in current_entry['divisions']:
                                current_entry['divisions'].append(div)
                        else:
                            # Collision?! Same teacher, different room/subject at same time?
                            # This implies a valid schedule issue, or maybe intentional parallel (extremely rare).
                            # We will note it.
                            current_entry['batches'].append(f"{batch} (CONFLICT: {subject} in {room})")
    
    # 2. Post-process to mark FREE slots
    # And convert list to a more friendly JSON structure if needed, or keep as arrays.
    # Arrays are good for PDF generation.
    
    # We will finalize the output dict
    final_output = {}
    
    for teacher, day_data in faculty_schedule.items():
        final_output[teacher] = {}
        for day in days:
            slots_list = []
            for i in range(total_slots):
                time_str = f"{start_hour + i}:00 - {start_hour + i + 1}:00"
                
                entry = day_data[day][i]
                if entry:
                    # Format for output
                    # Merge batches/divisions for display
                    current_batches = sorted(entry['batches'])
                    
                    # Check if it constitutes a full division
                    # We assume 4 batches per division based on config
                    # or strictly check validity
                    # Extract div names from batches? A1->A, B1->B
                    # Simpler: check if count is 4 (from config)
                    
                    batch_str = ""
                    if len(current_batches) == config.get('batches_per_division', 4):
                        # Use Division Name
                        # We have entry['divisions']
                        divs = entry.get('divisions', [])
                        if divs:
                            # entry['divisions'] might be ['DivA']
                            # User wants simply 'A' or 'D'.
                            # Let's strip "Div" from "DivA" -> "A"
                            clean_divs = [d.replace('Div', '') for d in divs]
                            batch_str = ", ".join(clean_divs)
                        else:
                             batch_str = "All"
                    else:
                        batch_str = ", ".join(current_batches)
                    
                    slots_list.append({
                        "time": time_str,
                        "status": "BUSY",
                        "subject": entry['subject'],
                        "room": entry['room'],
                        "type": entry['type'],
                        "batches": batch_str
                    })
                else:
                    # It's Free
                    slots_list.append({
                        "time": time_str,
                        "status": "FREE"
                    })
            final_output[teacher][day] = slots_list

    # Save
    with open('faculty_timetable.json', 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Generated faculty_timetable.json for {len(final_output)} teachers.")

if __name__ == "__main__":
    main()