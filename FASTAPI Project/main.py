"""
Building new project
"""


from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
items = []

@app.get("/")
def read_root():
    return {"message": "FASTAPI is working"}


@app.get("/hello")
def say_hello():
    return {"message:" "Hello from FASTAPI"}


class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True



@app.post("/items")
def create_item(item: Item):
    items.append(item.dict())
    return {"message": "Item added", "item": item}



@app.get("/items")
def get_items():
    return {"items": items}