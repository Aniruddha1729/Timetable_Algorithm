import json
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    rooms_data = load_json('rooms.json')
    timetable = load_json('timetable_full.json')
    config = load_json('config.json')

    all_rooms = list(rooms_data.keys())
    days = config['days']
    start_hour = config.get('start_hour', 9)
    # end_hour = config.get('end_hour', 17)
    # Assuming 8 slots based on standard day 9-5
    
    # Initialize availability: room -> day -> [True, True, ...] (True means Free)
    # We'll use a dictionary to track occupied slots. 
    # occupied[room][day][slot_index] = True/False
    
    occupied = {room: {day: [False] * 8 for day in days} for room in all_rooms}
    
    # Iterate through the timetable to mark occupied slots
    # Structure: Division -> Batch -> Day -> List of Slots
    for division, batches in timetable.items():
        for batch, day_schedule in batches.items():
            for day, slots in day_schedule.items():
                if day not in days:
                    continue
                
                for slot_index, slot_data in enumerate(slots):
                    # Check if slot_data has a 'room'
                    if 'room' in slot_data and slot_data['room']:
                        room = slot_data['room']
                        # Some consistency check: is this room in our rooms list?
                        if room in occupied:
                            occupied[room][day][slot_index] = True
                        else:
                            # It's possible a room is used that isn't in rooms.json, or format differs
                            # For safety, let's add it if missing, or just log it suitable for debugging
                            if room not in occupied:
                                occupied[room] = {d: [False] * 8 for d in days}
                                occupied[room][day][slot_index] = True

    # Now generate the free slots report
    free_slots = {}
    
    for room in all_rooms:
        free_slots[room] = {}
        for day in days:
            free_slots_indices = []
            for i in range(8):
                if not occupied[room][day][i]:
                    # Convert index to time string roughly
                    time_str = f"{start_hour + i}:00 - {start_hour + i + 1}:00"
                    free_slots_indices.append(time_str)
            
            if free_slots_indices:
                free_slots[room][day] = free_slots_indices
            else:
                free_slots[room][day] = []

    # Output to JSON
    with open('free_slots.json', 'w') as f:
        json.dump(free_slots, f, indent=4)
        
    print("Successfully generated 'free_slots.json'.")

    # print a small summary for the user
    print("\nSummary of Free Slots (First 5 rooms):")
    count = 0
    for room in free_slots:
        if count >= 5: break
        print(f"Room {room}:")
        for day, slots in free_slots[room].items():
            if slots:
                print(f"  {day}: {', '.join(slots)}")
        count += 1

if __name__ == "__main__":
    main()