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
    elements.append(Paragraph("Smart Timetable Comp Dept", title_style))
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
    # First col: Day Name (small)
    # Other cols: Slots (equal width)
    page_width = landscape(A4)[0] - 60 # margins
    col_width = (page_width - 0.8*inch) / len(slots)
    col_widths = [0.8*inch] + [col_width] * len(slots)
    
    # Process each Division
    for div_idx, (div_name, div_schedule) in enumerate(data.items()):
        if div_idx > 0:
            elements.append(PageBreak())
            
        elements.append(Paragraph(f"Division: {div_name}", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Prepare Table Data
        # Header Row
        headers = ['Day'] + slots
        table_data = [headers]
        
        # Rows are Days
        for day in days:
            row_data = [day]
            
            # For each slot, we need to gather info from ALL batches in this division
            batches = sorted(div_schedule.keys())
            
            for s_idx, slot_time in enumerate(slots):
                # Collect content for this slot across all batches
                batch_contents = {} # content -> list of batches
                
                for batch in batches:
                    # Get schedule for this batch, day
                    day_sched = div_schedule[batch].get(day, [])
                    # day_sched is a list of slot objects corresponding to slots indices
                    if s_idx < len(day_sched):
                        slot_info = day_sched[s_idx]
                        
                        if not slot_info:
                            content = "-"
                        elif slot_info == "-":
                            content = "-"
                        elif slot_info.get('type') == 'LUNCH':
                            content = "LUNCH"
                        elif slot_info.get('type') == 'FREE':
                             content = slot_info.get('class', 'Free')
                        else:
                            cls = slot_info.get('class', '')
                            teacher = slot_info.get('teacher', '')
                            room = slot_info.get('room', '')
                            # Format: "Subject (Teacher, Room)"
                            content = f"{cls}\n({teacher}  {room})"
                    else:
                        content = "-"
                        
                    if content not in batch_contents:
                        batch_contents[content] = []
                    batch_contents[content].append(batch)
                
                # Now format the cell content based on grouping
                cell_text_parts = []
                is_lunch = False
                is_lecture = False
                is_lab = False
                
                # Check if all batches have same content
                if len(batch_contents) == 1:
                    content = list(batch_contents.keys())[0]
                    if content == "LUNCH":
                        cell_text = "LUNCH"
                        is_lunch = True
                    elif content == "-" or content == "Free" or content == "<--Free-->":
                         cell_text = content
                    else:
                        # Single content for all (Lecture)
                        cell_text = content
                        is_lecture = True
                else:
                    # Mixed content (Labs or some free/some busy)
                    # We list them: "A1, A2: Content1 / A3: Content2"
                    # Sort by batch name to be deterministic
                    sorted_contents = []
                    for content, batch_list in batch_contents.items():
                        if content == "-" or content == "Free" or content == "<--Free-->":
                            continue # Skip empty/free in the merged display to save space? Or show "A1: Free"?
                            # User might want to see who is free. Let's include it if it's mixed.
                            # But usually we focus on classes.
                            # Let's include it for completeness but maybe simplify.
                        
                        batch_str = ", ".join(batch_list)
                        sorted_contents.append(f"{batch_str}: {content}")
                    
                    if not sorted_contents:
                        cell_text = "-"
                    else:
                        cell_text = "\n\n".join(sorted_contents)
                        is_lab = True # Assume mixed means lab/parallel sessions
                
                row_data.append(cell_text)
            
            table_data.append(row_data)
        
        # Create Table
        # We need to calculate row heights dynamically or allow them to grow
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
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ])
        
        # Apply conditional background colors
        # Iterate over the data we just built (skipping header)
        for r_idx, row in enumerate(table_data[1:], start=1):
            for c_idx, cell_content in enumerate(row[1:], start=1):
                # Check content to determine color
                if cell_content == "LUNCH":
                    style.add('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), colors.black)
                elif cell_content == "-" or cell_content == "Free" or cell_content == "<--Free-->":
                    pass # White
                elif ":" in cell_content:
                    # This indicates grouped batches -> Lab/Parallel
                    style.add('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), colors.lightgreen)
                else:
                    # Single content -> Lecture
                    style.add('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), colors.lightblue)

        # Horizontal Merging for Adjacent Identical Cells
        # Iterate through each row (Day)
        for r_idx in range(1, len(table_data)):
            row = table_data[r_idx]
            # Iterate through columns (Slots)
            # We start from col 1 (first slot)
            start_col = 1
            while start_col < len(row):
                content = row[start_col]
                
                # Don't merge empty/free cells if not desired, but user asked for "labs".
                # Labs usually have content.
                # Also don't merge if content is "-" or "Free" unless user wants that too.
                # Let's assume we merge anything identical except maybe LUNCH (which is already a column).
                # Actually, LUNCH is identical across all rows at the same column, so horizontal merge isn't needed for LUNCH (it's 1 hour).
                
                if content == "LUNCH":
                    start_col += 1
                    continue
                
                end_col = start_col
                # Check next columns
                while end_col + 1 < len(row):
                    next_content = row[end_col + 1]
                    if next_content == content and content != "-" and content != "Free" and content != "<--Free-->":
                        end_col += 1
                    else:
                        break
                
                if end_col > start_col:
                    # Merge from start_col to end_col
                    style.add('SPAN', (start_col, r_idx), (end_col, r_idx))
                    # Background color is already set for individual cells, SPAN will use the top-left style?
                    # ReportLab SPAN keeps the style of the first cell usually.
                
                start_col = end_col + 1

        t.setStyle(style)
        elements.append(t)
        elements.append(Spacer(1, 20))
            
    doc.build(elements)
    print(f"PDF generated successfully: {output_path}")

if __name__ == "__main__":
    generate_pdf()
