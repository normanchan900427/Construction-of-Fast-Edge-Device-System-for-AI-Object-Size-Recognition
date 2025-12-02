
import os, cv2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import tensormask
from tensormask import add_tensormask_config
import torch
from torch.utils.cpp_extension import CUDA_HOME 
####register
import register
####
import json


#TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
#CUDA_VERSION = torch.__version__.split("+")[-1]
#print("toch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
#print(torch.cuda.is_available(), CUDA_HOME)
my_metadata = MetadataCatalog.get("my_train")

if __name__ == "__main__":
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    add_tensormask_config(cfg)
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.merge_from_file("configs/tensormask_R_50_FPN_1x.yaml")
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "training_output/tmask_coco"

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    
    #cfg.MODEL.WEIGHTS = "C:/Users/Yi Wei Chan/detectron2-main/output/model_final.pth"
    #######print('loading from: {}'.format(cfg.MODEL.WEIGHTS))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.TENSOR_MASK.NMS_THRESH_TEST = 0.75
    cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.075
    #cfg.DATASETS.TEST = ()

    predictor = DefaultPredictor(cfg)
####################
    dataset_test = DatasetCatalog.get("my_test")
    for d in dataset_test:
        img = cv2.imread(d["file_name"])
        ####get the image file name
        img_name = d["file_name"].split("\\")
        img_name = img_name[len(img_name)-1]
        outputs = predictor(img)
        v = Visualizer(img[:], metadata=my_metadata, scale=1.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ####
        ####visualize prediction
        cv2.imshow("image",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        break
        
###############    

    #TEMP_ROOT = r'../detectron2-main/projects/TensorMask/database/static/dataset/test/3BF_20230412_0112_000397_5_jpg.rf.1fabaa7ddd34979c375c288c5c824a65.jpg'
    #im = cv2.imread(TEMP_ROOT)
    #outputs = predictor(im)

    #v = Visualizer(im[:, :, ::-1], metadata = my_metadata, scale=1.5)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    #cv2.imshow("image",out.get_image()[:, :, ::-1])
    #cv2.waitKey(0)                                                                                                                                  