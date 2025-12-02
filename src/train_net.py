#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TensorMask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from tensormask import add_tensormask_config

#####################################
from detectron2.data.build import build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T
#資料增量
augs = T.AugmentationList([
            T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
            T.RandomApply(T.RandomContrast(0.95, 1.05), prob=0.5),
            #T.ResizeShortestEdge((640, 800), 1333, "range"),
            T.RandomFlip(prob=0.5),
            T.RandomApply (
                T.RandomRotation(
                angle         = [-30, 30],
                sample_style  = "range",
                center        = [[0.4, 0.6], [0.4, 0.6]],
                expand        = False
            ), prob=0.5),
        ])
#####################################


class Trainer(DefaultTrainer):
    #########################
    #data augmentation
    #@classmethod
    #def build_train_loader(cls, cfg):
    #    custom_mapper = DatasetMapper(cfg, is_train=True, augmentations=[augs])
    #    return build_detection_train_loader(cfg, mapper=custom_mapper)
    #########################
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tensormask_config(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
