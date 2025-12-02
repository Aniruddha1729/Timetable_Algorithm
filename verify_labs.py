import json

with open('timetable_full.json', 'r') as f:
    data = json.load(f)

output_file = 'verify_labs_output.txt'
with open(output_file, 'w') as f:
    for div_name, div_schedule in data.items():
        print(f"\n{div_name} Analysis:", file=f)
        print(f"{'Batch':<10} {'Subject':<10} {'Count':<10} {'Slots'}", file=f)
        print("-" * 50, file=f)

        for batch, schedule in div_schedule.items():
            dsl_count = 0
            mal_count = 0
            devl_count = 0
            
            dsl_slots = [] 
            mal_slots = []
            devl_slots = []
            
            for day, slots in schedule.items():
                for i, slot in enumerate(slots):
                    cls = slot.get('class')
                    if cls == 'DSL':
                        dsl_count += 1
                        dsl_slots.append(f"{day} Slot {i+1}")
                    elif cls == 'MAL':
                        mal_count += 1
                        mal_slots.append(f"{day} Slot {i+1}")
                    elif cls == 'DEVL':
                        devl_count += 1
                        devl_slots.append(f"{day} Slot {i+1}")
                        
            print(f"{batch:<10} {'DSL':<10} {dsl_count:<10} {', '.join(dsl_slots)}", file=f)
            print(f"{batch:<10} {'DEVL':<10} {devl_count:<10} {', '.join(devl_slots)}", file=f)
            print(f"{batch:<10} {'MAL':<10} {mal_count:<10} {', '.join(mal_slots)}", file=f)
            print("-" * 50, file=f)

print(f"Verification report saved to {output_file}")
