import glob

import numpy as np


class NoLabelDataset(object):
    def __init__(self, split='train', img_h=720, img_w=1280, max_lanes=None, root=None, img_ext='.jpg'):
        self.root = root
        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = img_w, img_h
        self.img_ext = img_ext
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # On NoLabelDataset, always force it
        self.max_lanes = max_lanes
        self.max_points = 1

    def get_img_heigth(self, path):
        return self.img_h

    def get_img_width(self, path):
        return self.img_w

    def get_metrics(self, lanes, idx):
        return [1] * len(lanes), [1] * len(lanes), None

    def load_annotations(self):
        self.annotations = []
        pattern = '{}/**/*{}'.format(self.root, self.img_ext)
        print('Looking for image files with the pattern', pattern)
        for file in glob.glob(pattern, recursive=True):
            self.annotations.append({'lanes': [], 'path': file})

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        return "", None

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
