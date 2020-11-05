import sys

import numpy as np

from lib.datasets.lane_dataset import LaneDataset

EXPS_DIR = 'experiments'


class Evaluator(object):
    def __init__(self, dataset, exp_dir, poly_degree=3):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            exp_dir: (str): write your description
            poly_degree: (int): write your description
        """
        self.dataset = dataset
        # self.predictions = np.zeros((len(dataset.annotations), dataset.max_lanes, 4 + poly_degree))
        self.predictions = None
        self.runtimes = np.zeros(len(dataset))
        self.loss = np.zeros(len(dataset))
        self.exp_dir = exp_dir
        self.new_preds = False

    def add_prediction(self, idx, pred, runtime):
        """
        Add a new prediction.

        Args:
            self: (todo): write your description
            idx: (todo): write your description
            pred: (todo): write your description
            runtime: (int): write your description
        """
        if self.predictions is None:
            self.predictions = np.zeros((len(self.dataset.annotations), pred.shape[1], pred.shape[2]))
        self.predictions[idx, :pred.shape[1], :] = pred
        self.runtimes[idx] = runtime
        self.new_preds = True

    def eval(self, **kwargs):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
        """
        return self.dataset.dataset.eval(self.exp_dir, self.predictions, self.runtimes, **kwargs)


if __name__ == "__main__":
    evaluator = Evaluator(LaneDataset(split='test'), exp_dir=sys.argv[1])
    evaluator.tusimple_eval()
