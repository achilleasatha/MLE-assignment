from typing import List

from fastapi import FastAPI

app = FastAPI()

@app.post("/infer")
def infer(products: List[Product]):
    #TODO: calculate predictons for given inputs and return the result
