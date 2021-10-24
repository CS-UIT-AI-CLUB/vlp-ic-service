from fastapi import APIRouter

from ..vlp.vlp import seq2seq_loader
from ..vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.pytorch_pretrained_bert.tokenization import BertTokenizer

router = APIRouter()

@router.get('/predict')
def predict():

    return {'message': 'OK'}
