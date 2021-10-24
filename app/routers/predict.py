from fastapi import APIRouter

from ..vlp.vlp import seq2seq_loader
from ..vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.pytorch_pretrained_bert.tokenization import BertTokenizer

import torch

router = APIRouter()

class Config():
    bert_model = 'bert-base-multilingual-cased'
    do_lower_case = False
    max_tgt_length = 20
    len_vis_input = 100
    new_segment_ids = True
    enable_butd = True

args = Config()

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case)

args.max_seq_length = args.max_tgt_length + \
    args.len_vis_input + 3  # +3 for 2x[SEP] and [CLS]
tokenizer.max_len = args.max_seq_length

seq2seq4decode = seq2seq_loader.Preprocess4Seq2seqPredict(list(
    tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
    max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,
    mode='s2s', len_vis_input=args.len_vis_input, enable_butd=args.enable_butd,
    region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix)

@router.get('/predict')
def predict():

    return {'message': 'OK'}
