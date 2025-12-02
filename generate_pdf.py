import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
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
    for div_idx, (div_name, div_schedule) in enumerate(data.items()):
        if div_idx > 0:
            elements.append(PageBreak())
            
        elements.append(Paragraph(f"Division: {div_name}", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Process each Day
        for i, day in enumerate(days):
            # Page Break after every 2 days (but not before the first one on a new div page)
            if i > 0 and i % 2 == 0:
                elements.append(PageBreak())
                # Optional: Repeat Division Header
                elements.append(Paragraph(f"Division: {div_name} (Cont.)", styles['Heading2']))
                elements.append(Spacer(1, 10))

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
            
            for b_idx, batch in enumerate(batches):
                row_data = [batch]
                day_schedule = div_schedule[batch].get(day, [])
                
                # Initialize row with empty strings
                slot_contents = ["-"] * len(slots)
                
                # Fill slots
                for s_idx, slot_info in enumerate(day_schedule):
                    if not slot_info: continue
                    
                    if slot_info == "-":
                        content = "-"
                    elif slot_info.get('type') == 'LUNCH':
                        content = "LUNCH"
                    elif slot_info.get('type') == 'FREE':
                        content = slot_info.get('class', 'Free')
                    else:
                        cls = slot_info.get('class', '')
                        teacher = slot_info.get('teacher', '')
                        room = slot_info.get('room', '')
                        t_type = slot_info.get('type', '')
                        
                        # No prefix "LEC:" or "LAB:" as per user preference
                        content = f"{cls}\n({teacher}, {room})"
                    
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
            # 1. Lunch Column
            if lunch_idx is not None:
                col_idx = lunch_idx + 1 # +1 for Batch column
                style.add('BACKGROUND', (col_idx, 0), (col_idx, -1), colors.black)
                style.add('SPAN', (col_idx, 1), (col_idx, -1)) # Merge lunch vertically across batches
            
            # 2. Lectures (Merge Vertically across all batches)
            # ... (existing vertical merge logic) ...
            for s_idx in range(len(slots)):
                col_idx = s_idx + 1
                
                # Check if this slot is a Lecture for the first batch
                first_batch_content = table_data[1][col_idx]
                
                # Logic: If content is not empty/lunch AND identical across all batches -> Merge
                if first_batch_content != "-" and first_batch_content != "LUNCH":
                    # Check if all other batches have the same content
                    is_common = True
                    for r_idx in range(2, len(batches) + 1):
                        if table_data[r_idx][col_idx] != first_batch_content:
                            is_common = False
                            break
                    
                    if is_common:
                        # Merge Vertically (It's a Lecture)
                        style.add('SPAN', (col_idx, 1), (col_idx, -1))
                        style.add('BACKGROUND', (col_idx, 1), (col_idx, -1), colors.lightblue)
                    else:
                        # If not common, it might be a Lab.
                        # We handle coloring here, but horizontal merging is done below.
                        for r_idx in range(1, len(batches) + 1):
                            cell_content = table_data[r_idx][col_idx]
                            if cell_content != "-" and cell_content != "LUNCH":
                                style.add('BACKGROUND', (col_idx, r_idx), (col_idx, r_idx), colors.lightgreen)

            # 3. Labs (Merge Horizontally within each batch)
            # Iterate through each batch row
            for r_idx in range(1, len(batches) + 1):
                # We iterate through columns to find contiguous blocks
                start_col = 1
                while start_col <= len(slots):
                    content = table_data[r_idx][start_col]
                    
                    # Only merge if it's a Lab (Green) or Lecture (if we wanted, but Lectures are vert merged)
                    # Actually, if it's a vertically merged lecture, horizontal merge might conflict or be redundant?
                    # But usually lectures are 1 hour. If a lecture is 2 hours, we might want horizontal merge too?
                    # User asked for "labs". Labs are the ones that are definitely 2 hours.
                    # And Labs are NOT vertically merged.
                    
                    # Check if this cell is part of a vertical merge?
                    # Hard to check style object directly.
                    # But we know if it's a lecture (common across all), we colored it blue.
                    # If it's a lab, we colored it green.
                    
                    # Let's check if it's identical to the next column
                    end_col = start_col
                    while end_col + 1 <= len(slots):
                        next_content = table_data[r_idx][end_col + 1]
                        if next_content == content and content != "-" and content != "LUNCH":
                            end_col += 1
                        else:
                            break
                    
                    if end_col > start_col:
                        # Found a block to merge
                        # Check if it's a Lecture (Blue) or Lab (Green)
                        # If it's a Lecture, it's already vertically merged. Can we merge horizontally too?
                        # ReportLab allows SPAN over a rectangle.
                        # But if we have multiple vertical spans side-by-side, merging them horizontally might be tricky.
                        # However, for Labs, they are NOT vertically merged. So horizontal merge is safe.
                        
                        # How to distinguish?
                        # Re-run the "is_common" check?
                        is_lecture = True
                        if content != "-" and content != "LUNCH":
                             for check_r in range(1, len(batches) + 1):
                                 if table_data[check_r][start_col] != content:
                                     is_lecture = False
                                     break
                        else:
                            is_lecture = False
                            
                        if not is_lecture:
                            # It's a Lab (or unique session). Merge Horizontally.
                            style.add('SPAN', (start_col, r_idx), (end_col, r_idx))
                            # Background is already set to green in the vertical loop above.
                    
                    start_col = end_col + 1

            t.setStyle(style)
            elements.append(t)
            elements.append(Spacer(1, 20))
            
    doc.build(elements)
    print(f"PDF generated successfully: {output_path}")

if __name__ == "__main__":
    generate_pdf()
