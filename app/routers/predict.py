from fastapi import APIRouter

from ..vlp.vlp import seq2seq_loader
from ..vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.pytorch_pretrained_bert.tokenization import BertTokenizer

router = APIRouter()

class Config():
    bert_model = 'bert-base-multilingual-cased'
    do_lower_case = False
    max_tgt_length = 20
    len_vis_input = 100

args = Config()

tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case)

args.max_seq_length = args.max_tgt_length + \
    args.len_vis_input + 3  # +3 for 2x[SEP] and [CLS]
tokenizer.max_len = args.max_seq_length

@router.get('/predict')
def predict():

    return {'message': 'OK'}
