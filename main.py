from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import traceback
import os

from timetable_solver import TimetableScheduler
from ortools.sat.python import cp_model

# 🔹 IMPORT PDF GENERATOR
from generate_pdf import generate_pdf

app = FastAPI()

# -----------------------------
# Solver wrapper
# -----------------------------
def generate_timetable(
    config_path="config.json",
    freq_path="frequencies.json",
    teacher_path="teachers.json",
    room_path="rooms.json",
    output_file="timetable_full.json"
):
    try:
        scheduler = TimetableScheduler(
            config_path=config_path,
            freq_path=freq_path,
            teacher_path=teacher_path,
            room_path=room_path
        )

        scheduler.build_model()
        status = scheduler.solve()

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            scheduler.export_solution(filename=output_file)
            return True

        return False

    except Exception as e:
        print("Solver error:", e)
        traceback.print_exc()
        return False


# -----------------------------
# API Payload Schema
# -----------------------------
class TimetablePayload(BaseModel):
    divisions: list
    lectures: list
    labs: list
    frequencies: dict
    teachers: dict
    rooms: dict
    settings: dict


# -----------------------------
# Generate Timetable JSON
# -----------------------------
@app.post("/generate")
def generate(payload: TimetablePayload):
    print("Received timetable generation request.")

    with open("config.json", "w") as f:
        json.dump({
            "divisions": payload.divisions,
            "lectures": payload.lectures,
            "labs": payload.labs,
            **payload.settings
        }, f, indent=2)

    with open("frequencies.json", "w") as f:
        json.dump(payload.frequencies, f, indent=2)

    with open("teachers.json", "w") as f:
        json.dump(payload.teachers, f, indent=2)

    with open("rooms.json", "w") as f:
        json.dump(payload.rooms, f, indent=2)

    success = generate_timetable()

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Timetable generation failed. Check constraints."
        )

    with open("timetable_full.json") as f:
        return {"timetable": json.load(f)}


# -----------------------------
# 🆕 Generate & Download PDF
# -----------------------------
@app.post("/generate-pdf")
def generate_pdf_api():
    """
    Assumes timetable_full.json already exists
    """
    if not os.path.exists("timetable_full.json"):
        raise HTTPException(
            status_code=400,
            detail="Timetable not generated yet"
        )

    output_pdf = "timetable.pdf"

    try:
        generate_pdf(
            json_path="timetable_full.json",
            output_path=output_pdf
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="PDF generation failed"
        )

    return FileResponse(
        output_pdf,
        media_type="application/pdf",
        filename="timetable.pdf"
    )


# -----------------------------
# Optional CLI run
# -----------------------------
if __name__ == "__main__":
    ok = generate_timetable()
    print("SUCCESS" if ok else "FAILED")
