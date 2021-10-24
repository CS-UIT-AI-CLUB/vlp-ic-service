from fastapi import FastAPI
from routers import check_cuda

app = FastAPI()

app.include_router(check_cuda.router)

@app.get('/')
def home():
    return {'message': 'Hello world'}