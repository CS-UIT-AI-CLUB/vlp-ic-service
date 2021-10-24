from fastapi import APIRouter
import torch

router = APIRouter()

@app.get('/check_cuda')
def check_cuda():
    if torch.cuda.is_available():
        return {'cuda': True}
    else:
        return {'cuda': False}