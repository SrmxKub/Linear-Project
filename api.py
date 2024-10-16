from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend import most_similar_perfumes
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:8000",  
    "http://127.0.0.1:8000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


class Perfume(BaseModel):
    scents: str
    base: list
    middle: list
    typ: str
    min_price: float
    max_price: float
    top_n: int

@app.post("/similar-perfumes")
def similar_perfumes(data: Perfume):
    
    data.base = ", ".join(data.base)
    data.middle = ", ".join(data.middle)
    result = most_similar_perfumes(data.scents, data.base, data.middle, data.typ, data.min_price, data.max_price, data.top_n)
    
    if result is None:
        return "No perfumes found for the given department"
    return result.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)