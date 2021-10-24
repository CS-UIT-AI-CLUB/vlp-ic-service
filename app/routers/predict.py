from fastapi import APIRouter

from ..vlp.vlp import seq2seq_loader
from ..vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
import glob

router = APIRouter()


class Config():
    bert_model = 'bert-base-multilingual-cased'
    do_lower_case = False
    max_tgt_length = 20
    len_vis_input = 100
    new_segment_ids = True
    enable_butd = True
    max_position_embeddings = 512
    config_path = None
    model_recover_path = 'app/vlp/checkpoints/model.bin'
    beam_size = 3
    length_penalty = 0
    forbid_duplicate_ngrams = None
    forbid_ignore_word = None
    ngram_size = 3
    min_len = None
    region_bbox_file = ''
    region_det_file_prefix = ''
    fp16 = False


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

# Prepare model
cls_num_labels = 2
type_vocab_size = 6 if args.new_segment_ids else 2
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
    ["[MASK]", "[SEP]"])

forbid_ignore_set = None
if args.forbid_ignore_word:
    w_list = []
    for w in args.forbid_ignore_word.split('|'):
        if w.startswith('[') and w.endswith(']'):
            w_list.append(w.upper())
        else:
            w_list.append(w)
    forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

# print('Load model...')
for model_recover_path in glob.glob(args.model_recover_path.strip()):
    model_recover = torch.load(model_recover_path)
    model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
                                                max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
                                                    state_dict=model_recover, num_labels=cls_num_labels, cache_dir='/mlcv/WorkingSpace/Personals/khiemltt/cache',
                                                type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
                                                search_beam_size=args.beam_size, length_penalty=args.length_penalty,
                                                eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                                forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
                                                enable_butd=args.enable_butd, len_vis_input=args.len_vis_input)
    del model_recover
print('Model loaded')

print('Load model to GPU')
if args.fp16:
    model.half()
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
torch.cuda.empty_cache()
model.eval()
print('Model is now on GPU')
    

@router.get('/predict')
def predict():
    is_ready = next(model.parameters()).is_cuda
    return {'is_ready': is_ready}
