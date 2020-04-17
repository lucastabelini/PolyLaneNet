import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from .elas import ELAS
from .llamas import LLAMAS
from .tusimple import TuSimple
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 split='train',
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        if dataset == 'tusimple':
            self.dataset = TuSimple(split=split, **kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(split=split, **kwargs)
        elif dataset == 'elas':
            self.dataset = ELAS(split=split, **kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()

        self.transform_annotations()
        self.img_h, self.img_w = img_size

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation

        self.normalize = normalize
        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes = zip(old_lanes, categories)
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes = np.ones((self.dataset.max_lanes, 1 + 2 + 2 * self.dataset.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.dataset.max_points):(3 + self.dataset.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'],
            'label': lanes,
            'old_anno': anno,
            'categories': [cat for _, cat in old_lanes]
        }

        return new_anno

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        print('Transforming annotations...')
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        print('Done.')

    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            img, label, _ = self.__getitem__(idx, transform=True)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)

        img_h, img_w, _ = img.shape

        # Draw label
        for i, lane in enumerate(label):
            if lane[0] == 0:  # Skip invalid lanes
                continue
            lane = lane[3:]  # remove conf, upper and lower positions
            xs = lane[:len(lane) // 2]
            ys = lane[len(lane) // 2:]
            ys = ys[xs >= 0]
            xs = xs[xs >= 0]

            # draw GT points
            for p in zip(xs, ys):
                p = (int(p[0] * img_w), int(p[1] * img_h))
                img = cv2.circle(img, p, 5, color=GT_COLOR, thickness=-1)

            # draw GT lane ID
            cv2.putText(img,
                        str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))

        if pred is None:
            return img

        # Draw predictions
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        matches, accs, _ = self.dataset.get_metrics(pred, idx)
        overlay = img.copy()
        for i, lane in enumerate(pred):
            if matches[i]:
                color = PRED_HIT_COLOR
            else:
                color = PRED_MISS_COLOR
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

            # draw class icon
            if cls_pred is not None and len(points) > 0:
                class_icon = self.dataset.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)

            # draw lane accuracy
            if len(points) > 0:
                cv2.putText(img,
                            '{:.2f}'.format(accs[i] * 100),
                            tuple(points[len(points) // 2] - 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=.75,
                            color=color)
        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx, transform=True):
        item = self.dataset[idx]
        img = cv2.imread(item['path'])
        label = item['label']
        if transform:
            line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings = self.transform(image=img, line_strings=line_strings)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)


def main():
    import torch
    from lib.config import Config
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = Config('config.yaml')
    train_dataset = cfg.get_dataset('train')
    for idx in range(len(train_dataset)):
        img = train_dataset.draw_annotation(idx)
        cv2.imshow('sample', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
