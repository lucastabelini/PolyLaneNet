import glob

import numpy as np


class NoLabelDataset(object):
    def __init__(self, split='train', img_h=720, img_w=1280, max_lanes=None, root=None, img_ext='.jpg'):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            split: (int): write your description
            img_h: (int): write your description
            img_w: (int): write your description
            max_lanes: (int): write your description
            root: (str): write your description
            img_ext: (str): write your description
        """
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
        """
        Get the heap heap

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        return self.img_h

    def get_img_width(self, path):
        """
        Get the width of the image.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        return self.img_w

    def get_metrics(self, lanes, idx):
        """
        Returns the metrics from lanes

        Args:
            self: (todo): write your description
            lanes: (str): write your description
            idx: (int): write your description
        """
        return [1] * len(lanes), [1] * len(lanes), None

    def load_annotations(self):
        """
        Load annotations.

        Args:
            self: (todo): write your description
        """
        self.annotations = []
        pattern = '{}/**/*{}'.format(self.root, self.img_ext)
        print('Looking for image files with the pattern', pattern)
        for file in glob.glob(pattern, recursive=True):
            self.annotations.append({'lanes': [], 'path': file})

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            exp_dir: (str): write your description
            predictions: (todo): write your description
            runtimes: (int): write your description
            label: (todo): write your description
            only_metrics: (bool): write your description
        """
        return "", None

    def __getitem__(self, idx):
        """
        Return an item with the given index.

        Args:
            self: (todo): write your description
            idx: (list): write your description
        """
        return self.annotations[idx]

    def __len__(self):
        """
        Returns the length of the annotations.

        Args:
            self: (todo): write your description
        """
        return len(self.annotations)
