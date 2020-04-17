import os
import math
import random

import cv2
import numpy as np
import xmljson
from scipy import interpolate
from lxml.etree import fromstring

SPLIT_DIRECTORIES = {
    'train': [
        "BR_S02", "GRI_S02", "ROD_S01", "ROD_S03", "VIX_S01", "VIX_S03", "VIX_S04", "VIX_S05", "VIX_S06", "VIX_S07",
        "VIX_S08", "VIX_S09", "VIX_S10", "VV_S01", "VV_S03"
    ],
    'test': ["ROD_S02", "VV_S02", "VV_S04", "BR_S01", "GRI_S01", "VIX_S02", "VIX_S11"],
}

CATEGORY_TO_ID = {str(i): i + 1 for i in range(8)}
ID_TO_CATEGORY = {i + 1: str(i) for i in range(8)}


class ELAS(object):
    def __init__(self, split='train', max_lanes=None, root=None):
        self.root = root
        self.split = split
        if root is None:
            raise Exception('Please specify the root directory')

        if split not in SPLIT_DIRECTORIES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        self.anno_directories = SPLIT_DIRECTORIES[split]

        self.img_w, self.img_h = 640, 480
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

        self.class_icons = {
            cls_id: cv2.imread(os.path.join(self.root, 'lmt', 'type_{}.png'.format(cls_id)))
            for cls_id in ID_TO_CATEGORY
        }

    def get_class_icon(self, cls_id):
        return self.class_icons[cls_id]

    def get_img_heigth(self, path):
        return self.img_h

    def get_img_width(self, path):
        return self.img_w

    def get_metrics(self, lanes, idx):
        # Placeholders
        return [1] * len(lanes), [1] * len(lanes), None

    def interp_lane(self, lane, ys, step=10):
        pts = [[x, ys[i]] for i, x in enumerate(lane) if not math.isnan(float(x))]
        if len(pts) <= 1:
            return None
        spline = interpolate.splrep([pt[1] for pt in pts], [pt[0] for pt in pts], k=len(pts) - 1)
        interp_ys = list(range(min([pt[1] for pt in pts]), max([pt[1] for pt in pts]), step))
        interp_xs = interpolate.splev(interp_ys, spline)

        return list(zip(interp_xs, interp_ys))

    def load_dir_annotations(self, dataset_dir):
        annotations = []
        max_points = 0
        max_lanes = 0

        # read config.xml
        config_fname = os.path.join(dataset_dir, 'config.xml')
        if not os.path.isfile(config_fname):
            raise Exception('config.xml not found: {}'.format(config_fname))
        with open(config_fname, 'r') as hf:
            config = xmljson.badgerfish.data(fromstring(hf.read()))['config']

        # read ground truth
        gt_fname = os.path.join(dataset_dir, 'groundtruth.xml')
        if not os.path.isfile(gt_fname):
            raise Exception('groundtruth.xml not found: {}'.format(gt_fname))
        with open(gt_fname, 'r') as hf:
            gt = xmljson.badgerfish.data(fromstring(hf.read()))['groundtruth']

        # read frame annotations
        for frame in gt['frames']['frame']:
            img_fname = os.path.join(dataset_dir, 'images/lane_{}.png'.format(frame['@id']))

            y, h = config['dataset']['region_of_interest']['@y'], config['dataset']['region_of_interest']['@height']
            ys = [y, math.ceil(y + h / 4.), math.ceil(y + h / 2.), y + h - 1]
            pts = ['p1', 'p2', 'p3', 'p4']
            lanes = []
            categories = []
            for side in ['Left', 'Right']:
                lane = [frame['position'][side.lower()][pt]['$'] for pt in pts]
                lane = self.interp_lane(lane, ys)
                if lane is None:
                    continue
                max_points = max(max_points, len(lane))
                lanes.append(lane)
                category = str(frame['@lmt{}'.format(side)])
                categories.append(CATEGORY_TO_ID[category.split(';')[0]])
            max_lanes = max(max_lanes, len(lanes))
            annotations.append({'lanes': lanes, 'path': img_fname, 'categories': categories})

        return annotations, max_points, max_lanes

    def load_annotations(self):
        self.annotations = []
        self.max_points = 0
        self.max_lanes = 0
        for directory in self.anno_directories:
            dir_path = os.path.join(self.root, directory)
            dir_annos, dir_max_points, dir_max_lanes = self.load_dir_annotations(dir_path)

            self.annotations.extend(dir_annos)
            self.max_points = max(self.max_points, dir_max_points)
            self.max_lanes = max(self.max_lanes, dir_max_lanes)

        print('{} annotations found. max_points: {} | max_lanes: {}'.format(len(self.annotations), self.max_points,
                                                                            self.max_lanes))
        if self.split == 'train':
            random.shuffle(self.annotations)

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        # Placeholder
        return "", None

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
