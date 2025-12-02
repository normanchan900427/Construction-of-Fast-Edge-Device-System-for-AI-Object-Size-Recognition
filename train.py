# train test
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

import os
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from tensormask import add_tensormask_config
from detectron2 import model_zoo
#register
import register

if __name__ == "__main__":
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("configs/tensormask_R_50_FPN_1x.yaml")
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.OUTPUT_DIR = "training_output/tmask_coco"
    ##############pretrain
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = "../detectron2-main/output/pretrained.pth"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "pretrained.pth")
    cfg.SOLVER.CHECKPOINT_PERIOD = 400 # save model
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.STEPS = (200, 380)#　STEPS: (480000, 520000)
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000 #每張圖片可以偵測的物件最大值
    #PATIENCE = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    trainer.train()
    

