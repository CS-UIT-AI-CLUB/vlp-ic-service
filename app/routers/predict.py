import json
from fastapi import APIRouter, UploadFile, File

from ..vlp.vlp.loader_utils import batch_list_to_batch_tensors
from ..vlp.vlp import seq2seq_loader
from ..vlp.pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from ..vlp.pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
import glob
import numpy as np
import h5py
import os
import requests

router = APIRouter()


def post_process(cap):
    if cap.endswith(' .'):
        cap = cap[:-2] + '.'
    cap = cap.replace('X - quang', 'X-quang')
    cap = cap.replace('x - quang', 'x-quang')
    cap = cap.replace(' ,', ',')
    cap = cap.replace(' :', ':')
    return cap

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

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

print('Load model...')
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


@router.post('/predict')
def predict(file: UploadFile = File(...)):
    # base_dir = '/mlcv/Databases/Imagecaption/Dataset/vietcap4h-train-test-aug/VLP_format/region_feat_gvd_wo_bgd'
    # img_id = 'train_00000001'
    # f_feat = h5py.File(os.path.join(base_dir,
    #                                 'feat_cls_1000/coco_detection_vg_100dets_vlp_checkpoint_trainval_feat001.h5'), 'r')
    # f_cls = h5py.File(os.path.join(base_dir,
    #                             'feat_cls_1000/coco_detection_vg_100dets_vlp_checkpoint_trainval_cls001.h5'), 'r')
    # f_bbox = h5py.File(os.path.join(base_dir,
    #                                 'raw_bbox/coco_detection_vg_100dets_vlp_checkpoint_trainval_bbox001.h5'), 'r')
    # region_feat_vec = np.array(f_feat[img_id])
    # region_cls_vec = np.array(f_cls[img_id])
    # region_bbox_vec = np.array(f_bbox[img_id])

    f = {'image': file.file.read()}
    result = requests.post(
        'http://detectron-vlp-api:5055/api/detectron_vlp', files=f)
    try:
        result = result.json()
    except json.JSONDecodeError:
        return {
            'code': '1305',
            'status': 'Cannot decode JSON response'
        }

    print(result['message'])

    # input2decode = seq2seq4decode(
    #    region_feat_vec, region_cls_vec, region_bbox_vec)

    # with torch.no_grad():
    #     batch = batch_list_to_batch_tensors([input2decode])
    #     batch = [t.to(device) for t in batch]

    #     input_ids, token_type_ids, position_ids, input_mask, task_idx, img, vis_pe = batch

    #     if args.fp16:
    #         img = img.half()
    #         vis_pe = vis_pe.half()

    #     if args.enable_butd:
    #         conv_feats = img.data  # Bx100x2048
    #         vis_pe = vis_pe.data
    #     else:
    #         conv_feats, _ = cnn(img.data)  # Bx2048x7x7
    #         conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
    #                                      -1).permute(0, 2, 1).contiguous()

    #     traces = model(conv_feats, vis_pe, input_ids, token_type_ids,
    #                    position_ids, input_mask, task_idx=task_idx)
    #     if args.beam_size > 1:
    #         traces = {k: v.tolist() for k, v in traces.items()}
    #         output_ids = traces['pred_seq']
    #     else:
    #         output_ids = traces[0].tolist()
    #     # for i in range(len(buf)):
    #     w_ids = output_ids[0]
    #     output_buf = tokenizer.convert_ids_to_tokens(w_ids)
    #     output_tokens = []
    #     for t in output_buf:
    #         if t in ("[SEP]", "[PAD]"):
    #             break
    #         output_tokens.append(t)
    #     output_sequence = post_process(' '.join(detokenize(output_tokens)))

    return {
        'code': '1000',
        'status': 'Done',
        'data': {
            'caption': 'output_sequence'
        }
    }
