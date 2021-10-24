#!/bin/bash 

# Usage:
# ./extract_feat_flickr30k.sh [proc_split]
# where proc_split indicates the last a few digits of the image IDs.
# For example, proc_split can go from 000 to 999, when proc_split=000,
# images with name *000.jpg will be processed.
# Hence, you can submit 1000 jobs in parallel to extract the features
# for the entire dataset.

DATA_ROOT="/dataset/Dataset/vietcap4h-train-test-aug/VLP_format/"

python3 tools/extract_feature_bk.py \
    --featcls-output-dir $DATA_ROOT/VinVL_feature/region_feat_gvd_wo_bgd/feat_cls_1000 \
    --box-output-dir $DATA_ROOT/VinVL_feature/region_feat_gvd_wo_bgd/raw_bbox \
    --output-file-prefix coco_detection_vg_100dets_vlp_checkpoint_trainval \
    --max_bboxes 100 --min_bboxes 100 \
    --cfg e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.yaml \
    --wts e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl \
    --dataset COCO \
    $DATA_ROOT/images \
    | tee log/log_extract_features_vg_100dets_coco_"$1"
