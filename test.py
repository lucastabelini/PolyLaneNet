import os
import sys
import random
import logging
import argparse
import subprocess
from time import time

import cv2
import numpy as np
import torch

from lib.config import Config
from utils.evaluator import Evaluator


def test(model, test_loader, evaluator, exp_root, cfg, view, epoch, max_batches=None, verbose=True):
    if verbose:
        logging.info("Starting testing.")

    # Test the model
    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])

    model.eval()
    criterion_parameters = cfg.get_loss_parameters()
    test_parameters = cfg.get_test_parameters()
    criterion = model.loss
    loss = 0
    total_iters = 0
    test_t0 = time()
    loss_dict = {}
    with torch.no_grad():
        for idx, (images, labels, img_idxs) in enumerate(test_loader):
            if max_batches is not None and idx >= max_batches:
                break
            if idx % 1 == 0 and verbose:
                logging.info("Testing iteration: {}/{}".format(idx + 1, len(test_loader)))
            images = images.to(device)
            labels = labels.to(device)

            t0 = time()
            outputs = model(images)
            t = time() - t0
            loss_i, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            loss += loss_i.item()
            total_iters += 1
            for key in loss_dict_i:
                if key not in loss_dict:
                    loss_dict[key] = 0
                loss_dict[key] += loss_dict_i[key]

            outputs = model.decode(outputs, labels, **test_parameters)

            if evaluator is not None:
                lane_outputs, _ = outputs
                evaluator.add_prediction(img_idxs, lane_outputs.cpu().numpy(), t / images.shape[0])
            if view:
                outputs, extra_outputs = outputs
                preds = test_loader.dataset.draw_annotation(
                    idx,
                    pred=outputs[0].cpu().numpy(),
                    cls_pred=extra_outputs[0].cpu().numpy() if extra_outputs is not None else None)
                cv2.imshow('pred', preds)
                cv2.waitKey(0)

    if verbose:
        logging.info("Testing time: {:.4f}".format(time() - test_t0))
    out_line = []
    for key in loss_dict:
        loss_dict[key] /= total_iters
        out_line.append('{}: {:.4f}'.format(key, loss_dict[key]))
    if verbose:
        logging.info(', '.join(out_line))

    return evaluator, loss / total_iters


def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()


def get_code_state():
    state = "Git hash: {}".format(
        subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    state += '\n*************\nGit diff:\n*************\n'
    state += subprocess.run(['git', 'diff'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    return state


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size

    # Model
    model = cfg.get_model().to(device)
    test_epoch = args.epoch

    # Get data set
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size if args.view is False else 1,
                                              shuffle=False,
                                              num_workers=8)
    # Eval results
    evaluator = Evaluator(test_loader.dataset, exp_root)

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )
    logging.info('Code state:\n {}'.format(get_code_state()))
    _, mean_loss = test(model, test_loader, evaluator, exp_root, cfg, epoch=test_epoch, view=args.view)
    logging.info("Mean test loss: {:.4f}".format(mean_loss))

    evaluator.exp_name = args.exp_name

    eval_str, _ = evaluator.eval(label='{}_{}'.format(os.path.basename(args.exp_name), test_epoch))

    logging.info(eval_str)
