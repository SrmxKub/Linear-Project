from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend import most_similar_perfumes, get_heatmap
import uvicorn

app = FastAPI()

# Input schema for perfume similarity request
class Perfume(BaseModel):
    scents: str
    base_notes: str
    middle_notes: str
    department: str

@app.post("/similar-perfumes")
def similar_perfumes(data: Perfume):
    result = most_similar_perfumes(data.scents, data.base_notes, data.middle_notes, data.department)
    
    if result is None:
        return "No perfumes found for the given department"
    return result.to_dict(orient="records")

@app.get("/perfume-correlations/{department}")
def perfume_correlations(department: str):
    heatmap = get_heatmap(department)
    
    if heatmap is None:
        return "No perfumes found for the given department"
    return {"heatmap": heatmap}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
