import json

with open('timetable_full.json', 'r') as f:
    data = json.load(f)

with open('frequencies.json', 'r') as f:
    freq_config = json.load(f)

# Identify which subjects are lectures
# We can infer this from config.json or just check the "type" in the output
with open('config.json', 'r') as f:
    config = json.load(f)
    lecture_subjects = set(config.get('lectures', []))
 
output_file = 'verify_lectures_output.txt'
with open(output_file, 'w') as f:
    print("Lecture Frequency Verification:", file=f)
    print(f"{'Division':<10} {'Subject':<10} {'Expected':<10} {'Actual':<10} {'Status'}", file=f)
    print("-" * 60, file=f)

    for div_name, div_schedule in data.items():
        # Lectures are common for the division, so we can just check one batch (e.g., the first one)
        # OR we can iterate all batches and ensure they are consistent.
        # Let's check the first batch since lectures are division-level.
        first_batch = list(div_schedule.keys())[0]
        schedule = div_schedule[first_batch]
        
        actual_counts = {subj: 0 for subj in lecture_subjects}
        
        for day, slots in schedule.items():
            for slot in slots:
                if slot.get('type') == 'LECTURE':
                    subj = slot.get('class')
                    if subj in actual_counts:
                        actual_counts[subj] += 1
        
        for subj in lecture_subjects:
            if subj not in freq_config: continue
            expected = freq_config[subj]
            actual = actual_counts[subj]
            status = "OK" if expected == actual else "MISMATCH"
            print(f"{div_name:<10} {subj:<10} {expected:<10} {actual:<10} {status}", file=f)
        print("-" * 60, file=f)

print(f"Lecture verification report saved to {output_file}")
