#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""
Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
Modified by Tina Jiang
Again modified by Luowei Zhou on 12/18/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
#import datasets.dummy_datasets as dummy_datasets

from tools.demo.detect_utils import detect_objects_on_single_image
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.transforms import build_transforms
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.scene_parser import SceneParser

import os.path as op
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import numpy as np
import base64
import csv
import timeit
import json
import h5py
import itertools
from icecream import ic

from utils.io import cache_url
import utils.c2 as c2_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
#cv2.ocl.setUseOpenCL(False)

from caffe2.python import workspace
import caffe2

from core.config import assert_and_infer_cfg
#from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from utils.boxes import nms
c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

csv.field_size_limit(sys.maxsize)


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.2
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl',
        type=str
    )
    parser.add_argument(
        '--box-output-dir',
        dest='box_output_dir',
        help='bounding box output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--featcls-output-dir',
        dest='featcls_output_dir',
        help='region feature and class prob output dir name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--output-file-prefix',
        dest='output_file_prefix',
        required=True,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--min_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--max_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--feat_name',
        help=" the name of the feature to extract, default: gpu_0/fc6",
        type=str,
        default="gpu_0/fc6"
    )
    parser.add_argument(
        '--dataset',
        help='Support dataset COCO | CC | Flickr30k | SBU',
        type=str,
        default='Flickr30k'
    )
    parser.add_argument(
        '--proc_split',
        help='only process image IDs that match this pattern at the end',
        type=str,
        default='000'
    )
    parser.add_argument(
        '--data_type',
        help='default float32, set to float16 to save storage space (e.g., for CC and SBU)',
        type=str,
        default='float32'
    )
    parser.add_argument(
        '--im_or_folder',
        help='image or folder of images',
        default=None
    )
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the proceed image")
    parser.add_argument("--visualize_attr", action="store_true",
                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    return parser.parse_args()


def get_detections_from_im(cfg, model, im, image_id, featmap_blob_name, feat_blob_name ,MIN_BOXES, MAX_BOXES, conf_thresh=0.2, bboxes=None):

    assert conf_thresh >= 0.
    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(model, im,cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=bboxes)
        num_rpn = scores.shape[0]
        region_feat = workspace.FetchBlob(feat_blob_name)
        max_conf = np.zeros((num_rpn,), dtype=np.float32)
        max_cls = np.zeros((num_rpn,), dtype=np.int32)
        max_box = np.zeros((num_rpn, 4), dtype=np.float32)

        for cls_ind in range(1, cfg.MODEL.NUM_CLASSES):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes[:, (cls_ind*4):(cls_ind*4+4)], cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            inds_update = np.where(cls_scores[keep] > max_conf[keep])
            kinds = keep[inds_update]
            max_conf[kinds] = cls_scores[kinds]
            max_cls[kinds] = cls_ind
            max_box[kinds] = dets[kinds][:,:4]

        keep_boxes = np.where(max_conf > conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        objects = max_cls[keep_boxes]
        obj_prob = max_conf[keep_boxes]
        obj_boxes = max_box[keep_boxes, :]
        cls_prob = scores[keep_boxes, :]

    # print('{} ({}x{}): {} boxes, box size {}, feature size {}, class size {}'.format(image_id,
    #       np.size(im, 0), np.size(im, 1), len(keep_boxes), cls_boxes[keep_boxes].shape,
    #       box_features[keep_boxes].shape, objects.shape))
    # print(cls_boxes[keep_boxes][:10, :], objects[:10], obj_prob[:10])

    assert(np.sum(objects>=cfg.MODEL.NUM_CLASSES) == 0)
    # assert(np.min(obj_prob[:10])>=0.2)
    # if np.min(obj_prob) < 0.2:
        # print('confidence score too low!', np.min(obj_prob[:10]))
    # if np.max(cls_boxes[keep_boxes]) > max(np.size(im, 0), np.size(im, 1)):
    #     print('box is offscreen!', np.max(cls_boxes[keep_boxes]), np.size(im, 0), np.size(im, 1))

    return {
        "image_id": image_id,
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': obj_boxes,
        'region_feat': region_feat[keep_boxes, :],
        'object': objects,
        'obj_prob': obj_prob,
        'cls_prob': cls_prob
    }


def get_detections_from_img_vinvl(cfg, model, model_ft, im, image_id, transforms, feat_blob_name, conf_thresh=0.2):
    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    try:
        visual_labelmap = load_labelmap_file(args.labelmap_file)
    except:
        visual_labelmap = None


    dets = detect_objects_on_single_image(model, transforms, im)

    if isinstance(model, SceneParser):
        rel_dets = dets['relations']
        dets = dets['objects']

    # for obj in dets:
    #     obj["class"] = dataset_labelmap[obj["class"]]
    if visual_labelmap is not None:
        dets = [d for d in dets if d['class'] in visual_labelmap]
    # if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
    #     for obj in dets:
    #         obj["attr"], obj["attr_conf"] = postprocess_attr(
    #             dataset_attr_labelmap, obj["attr"], obj["attr_conf"])
    # if cfg.MODEL.RELATION_ON and args.visualize_relation:
    #     for rel in rel_dets:
    #         rel['class'] = dataset_relation_labelmap[rel['class']]
    #         subj_rect = dets[rel['subj_id']]['rect']
    #         rel['subj_center'] = [
    #             (subj_rect[0] + subj_rect[2]) / 2, (subj_rect[1] + subj_rect[3]) / 2]
    #         obj_rect = dets[rel['obj_id']]['rect']
    #         rel['obj_center'] = [
    #             (obj_rect[0] + obj_rect[2]) / 2, (obj_rect[1] + obj_rect[3]) / 2]

    rects = [d["rect"] for d in dets]
    scores = [d["conf"] for d in dets]
    labels = [d["class"] for d in dets]

    len_rects, len_scores, len_labels = len(rects), len(scores), len(labels)

    assert conf_thresh >= 0.
    with c2_utils.NamedCudaScope(0):
        scores_tmp, cls_boxes, _ = infer_engine.im_detect_bbox(
            model_ft, im, 800, 1333, boxes=None)
        num_rpn = scores_tmp.shape[0]
        region_feat = workspace.FetchBlob(feat_blob_name)
        max_conf = np.zeros((num_rpn,), dtype=np.float32)
        max_cls = np.zeros((num_rpn,), dtype=np.int32)
        max_box = np.zeros((num_rpn, 4), dtype=np.float32)

        for cls_ind in range(1, 1601):
            cls_scores = scores_tmp[:, cls_ind]
            dets = np.hstack(
                (cls_boxes[:, (cls_ind * 4):(cls_ind * 4 + 4)], cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, 0.3))
            inds_update = np.where(cls_scores[keep] > max_conf[keep])
            kinds = keep[inds_update]
            max_conf[kinds] = cls_scores[kinds]
            max_cls[kinds] = cls_ind
            max_box[kinds] = dets[kinds][:, :4]

        keep_boxes = np.where(max_conf > conf_thresh)[0]
        if len(keep_boxes) < 100:
            keep_boxes = np.argsort(max_conf)[::-1][:100]
        elif len(keep_boxes) > 100:
            keep_boxes = np.argsort(max_conf)[::-1][:100]

        objects = max_cls[keep_boxes]
        obj_prob = max_conf[keep_boxes]
        obj_boxes = max_box[keep_boxes, :]
        cls_prob = scores_tmp[keep_boxes, :]

    rects.extend([[0] * 4] * (100 - len_rects))
    labels.extend([0] * (100 - len_labels))
    scores.extend([0] * (100 - len_scores))

    return {
            "image_id": image_id,
            "image_h": np.size(im, 0),
            "image_w": np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'boxes': np.array(rects, dtype=np.float32),
            'region_feat': region_feat[keep_boxes, :],
            'object': np.array(labels, dtype=np.int32),
            'obj_prob': np.array(scores, dtype=np.float32),
            'cls_prob': np.array(scores, dtype=np.float32)
    }


def main(args):
    logger = logging.getLogger(__name__)
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    merge_cfg_from_file(args.cfg)
    assert_and_infer_cfg(cache_urls=False)
    model_ft = infer_engine.initialize_model_from_cfg(args.weights)

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    start = timeit.default_timer()
    # extract region bboxes and features from pre-trained models
    count = 0
    if not os.path.exists(args.box_output_dir):
        os.makedirs(args.box_output_dir)
    if not os.path.exists(args.featcls_output_dir):
        os.makedirs(args.featcls_output_dir)

    if os.path.isdir(args.im_or_folder):
        img_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)


    for proc_split in range(0, 1000):
        proc_split = '0' * (3 - len(str(proc_split))) + str(proc_split)
        print('Proc split {}'.format(proc_split))
        with h5py.File(os.path.join(args.box_output_dir, args.output_file_prefix+'_bbox'+proc_split+'.h5'), "w") as f, \
            h5py.File(os.path.join(args.featcls_output_dir, args.output_file_prefix+'_feat'+proc_split+'.h5'), "w") as f_feat, \
            h5py.File(os.path.join(args.featcls_output_dir, args.output_file_prefix+'_cls'+proc_split+'.h5'), "w") as f_cls:

            if args.dataset in ('COCO', 'Flickr30k'):
                if os.path.isdir(args.im_or_folder):
                    im_list = glob.iglob(
                        args.im_or_folder + '/*.' + args.image_ext)

            for i, im_name in enumerate(im_list):
                im_base_name = os.path.basename(im_name)
                image_id = im_base_name

                if image_id[-4 - len(proc_split):-4] == proc_split:
                    im_name = os.path.join(args.im_or_folder, image_id)
                    ic(im_name)

                    transforms = build_transforms(cfg, is_train=False)
                    im = cv2.imread(im_name)
                    result = get_detections_from_img_vinvl(cfg, model, model_ft, im, image_id, transforms, args.feat_name)
                    # store results
                    proposals = np.concatenate((result['boxes'], np.expand_dims(result['object'], axis=1) \
                        .astype(np.float32), np.expand_dims(result['obj_prob'], axis=1)), axis=1)
                    
                    # ic(result, proposals,
                    #    result['region_feat'].shape, proposals.shape)

                    # ic(result, proposals,
                    #    result['region_feat'].shape, proposals.shape)

                    f.create_dataset(image_id[:-4], data=proposals.astype(args.data_type))
                    f_feat.create_dataset(image_id[:-4], data=result['region_feat'].squeeze().astype(args.data_type))
                    f_cls.create_dataset(image_id[:-4], data=result['cls_prob'].astype(args.data_type))

                    p = '/dataset/Dataset/vietcap4h-public-test/VLP_format/VinVL_feature_new/region_feat_gvd_wo_bgd/images'
                    if not os.path.exists(p):
                        os.makedirs(p)
                    feat_file = os.path.join(p, image_id[:-4] + '.npy')
                    np.save(feat_file, im)

                    count += 1
                    if count % 10 == 0:
                        end = timeit.default_timer()
                        epoch_time = end - start
                        print('process {:d} images after {:.1f} s'.format(count, epoch_time))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
