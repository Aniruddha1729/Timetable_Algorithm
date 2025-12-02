from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from algorithm.timetable_solver import generate_timetable

app = FastAPI()

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(data: dict):
    """
    Receives JSON from Next.js:
    {
      "config": {...},
      "frequencies": {...},
      "teachers": {...},
      "rooms": {...}
    }
    """
    timetable = generate_timetable(data)
    print("Generated Timetable:", timetable)
    return {"timetable": timetable}
