####mask二值圖轉座標
import numpy as np
import imantics
from imantics import Polygons, Mask

import os, cv2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
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
my_metadata = MetadataCatalog.get("my_train")

if __name__ == "__main__":
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("configs/tensormask_R_50_FPN_1x.yaml")
    cfg.OUTPUT_DIR = "training_output/tmask_coco"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.TENSOR_MASK.NMS_THRESH_TEST = 0.75
    cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.075
    predictor = DefaultPredictor(cfg)
####################
    ####save to coco json
    cocoData = {
        "info": {
            "year": '2023',
            "version": '3',
            "description": '',
            "contributor": '',
            "url": '',
            "date_created": '2023-07-31T03:37:35+00:00',
            },
        "licenses": [
            {
            "id": 1,
            "url": '',
            "name": 'CC BY 4.0',
            },
        ],
        "categories": [
            {
            "id": 0,
            "name": register.CLASS_NAMES,    ####get class name
            "supercategory": 'none',
            },
        ],
        "images" : [],
        "annotations" : [],
    }
    dataset_test = DatasetCatalog.get("my_test")
    img_id = 0
    obj_id = 0
    for d in dataset_test:
        img = cv2.imread(d["file_name"])
        ####get the image file name
        img_name = d["file_name"].split("\\")
        img_name = img_name[len(img_name)-1]
        ####predict
        outputs = predictor(img)
        img_info =   {
            "id": img_id,
            "license": 1,
            "file_name": img_name,
            "height": 512,
            "width": 512,
            "date_captured": '2023-07-31T03:37:35+00:00',
        }
        cocoData["images"].append(img_info)
        for i in range(0,len(outputs['instances'])):
            
            array = outputs['instances'][i].pred_masks.to('cpu').numpy()
            #print(len(outputs['instances']))

            height = array.shape[1]
            width = array.shape[2]
            array = array.reshape(height,width)


            polygons = Mask(array).polygons()
            mask = Mask.from_polygons(polygons)
            # get object area
            area = mask.area()
            # get bbox
            bbox = mask.bbox()
            obj = {
                "id" : obj_id,
                "image_id" : img_id,
                "category_id" : 0,
                "bbox" : list(bbox),
                "area" : float(area),
                "segmentation" : list(polygons.segmentation),
                "iscrowd" : 0
            }
            cocoData['annotations'].append(obj)
            obj_id += 1
            #print(polygons.points)
            #print("id: ",i,polygons.segmentation)
        img_id += 1
        #break
    #print(cocoData)
    with open("../detectron2-main/projects/TensorMask/output.coco.json", "w") as f:
        json.dump(cocoData, f, indent = 4)