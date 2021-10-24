from fastapi import FastAPI
from .routers import check_cuda, predict
from ..vlp.vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.vlp import seq2seq_loader

app = FastAPI()

app.include_router(predict.router)
app.include_router(check_cuda.router)

@app.get('/')
def home():
    return {'message': 'Hello world'}
