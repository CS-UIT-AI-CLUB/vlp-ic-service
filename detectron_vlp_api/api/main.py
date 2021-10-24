from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from utils.boxes import nms
import utils.vis as vis_utils
import utils.logging
import datasets.dummy_datasets as dummy_datasets
import core.test_engine as infer_engine
from utils.timer import Timer
from core.config import merge_cfg_from_file
from core.config import cfg
from core.config import assert_and_infer_cfg
import caffe2
from caffe2.python import workspace

from icecream import ic
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


from utils.io import cache_url
import utils.c2 as c2_utils
from tools.extract_feature_bk import get_detections_from_im

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

csv.field_size_limit(sys.maxsize)

from app import api, app, basedir
from flask_restful import Resource, reqparse
import werkzeug

import os
import math

import numpy as np

from utils_api.parser import get_config
cfg_service = get_config()
cfg_service.merge_from_file('cfg/service.yml')

# model_path and some related stuff
MODEL_PATH = cfg_service.SERVICE.CHECKPOINT_PATH
CFG_PATH = cfg_service.SERVICE.CONFIG_PATH
SERVICE_IP = cfg_service.SERVICE.SERVICE_IP
SERVICE_PORT = cfg_service.SERVICE.SERVICE_PORT
MAX_BBOXES = cfg_service.SERVICE.MAX_BBOXES
MIN_BBOXES = cfg_service.SERVICE.MIN_BBOXES
DATA_TYPE = cfg_service.SERVICE.TYPE
FEATURE_NAME = cfg_service.SERVICE.FEATURE_NAME

# allowed image extension
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg', 'bmp'])

def allowed_file(filename):
    	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
logger = logging.getLogger(__name__)
merge_cfg_from_file(CFG_PATH)
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(MODEL_PATH)

# Special json encoder for numpy types
class NumpyEncoder(json.JSONEncoder):  
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Service class
class serviceDetectronVLPHandler(Resource):
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files') 

		args = parser.parse_args()

		try:
			image_filename = args['image'].filename
			print(image_filename)
			if allowed_file(image_filename):
				image_file = args['image'].read()
				npimg = np.fromstring(image_file, np.uint8)
				img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
				cv2.imwrite(os.path.join(basedir, app.config['UPLOAD_FOLDER'], image_filename), img)

				result = get_detections_from_im(cfg, model, img, image_filename, '', FEATURE_NAME,
												MIN_BBOXES, MAX_BBOXES)

				# store results
				proposals = np.concatenate((result['boxes'], np.expand_dims(result['object'], axis=1)
											.astype(np.float32), np.expand_dims(result['obj_prob'], axis=1)), axis=1)
				results_dict = {}
				results_dict['proposals'] = proposals.astype(DATA_TYPE)
				results_dict['region_feat'] = result['region_feat'].squeeze().astype(DATA_TYPE)
				results_dict['cls_prob'] = result['cls_prob'].astype(DATA_TYPE)

				return {'message': 'Successfully', 'result': json.dumps(results_dict, cls=NumpyEncoder)}, 200
			else:
				return {'message': 'Not in allowed file'}, 420
		except:
			return {'message': 'No file selected'}, 419
		

api.add_resource(serviceDetectronVLPHandler, '/api/detectron_vlp')
if __name__ == "__main__":
    app.run(host=SERVICE_IP, debug=True, port=SERVICE_PORT)
