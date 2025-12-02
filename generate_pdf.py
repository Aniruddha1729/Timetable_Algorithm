import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf(json_path='timetable_full.json', output_path='timetable.pdf'):
    # Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    doc = SimpleDocTemplate(output_path, pagesize=landscape(A4), topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    # Title
    elements.append(Paragraph("SmartStack Timetable", title_style))
    elements.append(Spacer(1, 20))
    
    days = config['days']
    # Generate slots from start/end hour
    start_h = config['start_hour']
    end_h = config['end_hour']
    slots = [f"{h}:00" for h in range(start_h, end_h)]
    
    # Calculate lunch index
    lunch_start = config.get('lunch_start_hour')
    lunch_idx = None
    if lunch_start:
        try:
            lunch_idx = slots.index(f"{lunch_start}:00")
        except ValueError:
            pass
    
    # Define Column Widths
    # First col: Batch Name (small)
    # Other cols: Slots (equal width)
    # Total width approx 11 inches (landscape A4 is 11.69)
    page_width = landscape(A4)[0] - 60 # margins
    col_width = (page_width - 0.8*inch) / len(slots)
    col_widths = [0.8*inch] + [col_width] * len(slots)
    
    # Process each Division
    for div_name, div_schedule in data.items():
        elements.append(Paragraph(f"Division: {div_name}", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Process each Day
        for day in days:
            elements.append(Paragraph(f"<b>{day}</b>", styles['Heading3']))
            
            # Prepare Table Data
            # Header Row
            headers = ['Batch'] + slots
            table_data = [headers]
            
            # Batch Rows
            batches = sorted(div_schedule.keys())
            
            # We need to handle merging for lectures.
            # Strategy: Build the full grid first, then identify merge ranges.
            # Grid: rows = batches, cols = slots
            
            grid = {} # (batch_idx, slot_idx) -> content
            
            for b_idx, batch in enumerate(batches):
                row_data = [batch]
                day_schedule = div_schedule[batch].get(day, [])
                
                # Initialize row with empty strings
                slot_contents = ["-"] * len(slots)
                
                # Fill slots
                # Note: day_schedule is a list of dicts, but we need to map to slot indices.
                # The JSON structure is: "Mon": [{"type":..., "class":...}, ...]
                # Wait, the JSON output is a list of objects corresponding to slots?
                # Let's check the JSON structure again.
                # In timetable_solver.py:
                # schedule[day] = [ {slot_info} for s in slots ]
                # So it's a direct list mapping 1-to-1 with slots.
                
                for s_idx, slot_info in enumerate(day_schedule):
                    if not slot_info: continue
                    
                    if slot_info == "-":
                        content = "-"
                    elif slot_info.get('type') == 'LUNCH':
                        content = "LUNCH"
                    else:
                        cls = slot_info.get('class', '')
                        teacher = slot_info.get('teacher', '')
                        room = slot_info.get('room', '')
                        t_type = slot_info.get('type', '')
                        
                        if t_type == 'LECTURE':
                            content = f"LEC: {cls}\n({teacher}, {room})"
                        else:
                            content = f"LAB: {cls}\n({teacher}, {room})"
                    
                    slot_contents[s_idx] = content
                
                row_data.extend(slot_contents)
                table_data.append(row_data)
            
            # Create Table
            t = Table(table_data, colWidths=col_widths)
            
            # Styling
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ])
            
            # Add Conditional Styling & Merging
            # 1. Lunch Column (usually index 4+1=5 if lunch is at index 4)
            # We need to find the lunch index.
            lunch_idx = config.get('lunch_slot_index')
            if lunch_idx is not None:
                col_idx = lunch_idx + 1 # +1 for Batch column
                style.add('BACKGROUND', (col_idx, 0), (col_idx, -1), colors.lightgrey)
                style.add('SPAN', (col_idx, 1), (col_idx, -1)) # Merge lunch vertically across batches?
                # Actually, merging vertically for lunch is good.
            
            # 2. Lectures (Merge Vertically across all batches)
            # If all batches have the same lecture at the same slot, merge them.
            # We check column by column.
            for s_idx in range(len(slots)):
                col_idx = s_idx + 1
                
                # Check if this slot is a Lecture for the first batch
                first_batch_content = table_data[1][col_idx]
                if "LEC:" in first_batch_content:
                    # Check if all other batches have the same content
                    is_common = True
                    for r_idx in range(2, len(batches) + 1):
                        if table_data[r_idx][col_idx] != first_batch_content:
                            is_common = False
                            break
                    
                    if is_common:
                        # Merge Vertically
                        style.add('SPAN', (col_idx, 1), (col_idx, -1))
                        style.add('BACKGROUND', (col_idx, 1), (col_idx, -1), colors.lightblue)
                elif "LAB:" in first_batch_content or "LAB:" in table_data[2][col_idx]:
                     # Color Labs Green (no merge usually, unless same lab?)
                     # We can color individual cells
                     for r_idx in range(1, len(batches) + 1):
                         cell_content = table_data[r_idx][col_idx]
                         if "LAB:" in cell_content:
                             style.add('BACKGROUND', (col_idx, r_idx), (col_idx, r_idx), colors.lightgreen)

            t.setStyle(style)
            elements.append(t)
            elements.append(Spacer(1, 20))
            
        elements.append(Spacer(1, 30)) # Space between divisions

    doc.build(elements)
    print(f"PDF generated successfully: {output_path}")

if __name__ == "__main__":
    generate_pdf()
