import os
import json
import random

import numpy as np
from tabulate import tabulate

from utils.lane import LaneEval
from utils.metric import eval_json

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TuSimple(object):
    def __init__(self, split='train', max_lanes=None, root=None, metric='default'):
        self.split = split
        self.root = root
        self.metric = metric

        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 1280, 720
        self.max_points = 0
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_img_heigth(self, path):
        return 720

    def get_img_width(self, path):
        return 1280

    def get_metrics(self, lanes, idx):
        label = self.annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, _, _, matches, accs, dist = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)

        return matches, accs, dist

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            if lane[0] == 0:
                continue
            lane_pred = np.polyval(lane[3:], ys) * self.img_w
            lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
            lanes.append(list(lane_pred))

        return lanes

    def load_annotations(self):
        self.annotations = []
        max_lanes = 0
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.max_points = max(self.max_points, max([len(l) for l in gt_lanes]))
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })

        if self.split == 'train':
            random.shuffle(self.annotations)
        print('total annos', len(self.annotations))
        self.max_lanes = max_lanes

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, runtimes, filename):
        lines = []
        for idx in range(len(predictions)):
            line = self.pred2tusimpleformat(idx, predictions[idx], runtimes[idx])
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        pred_filename = '/tmp/tusimple_predictions_{}.json'.format(label)
        self.save_tusimple_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        elif self.metric == 'ours':
            result = json.loads(eval_json(pred_filename, self.anno_files[0], json_type='tusimple'))
        table = {}
        for metric in result:
            table[metric['name']] = [metric['value']]
        table = tabulate(table, headers='keys')

        if not only_metrics:
            filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)

        return table, result

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
