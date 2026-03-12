from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "FASTAPI is working"}


@app.get("/hello")
def say_hello():
    return {"message:" "Hello from FASTAPI"}


