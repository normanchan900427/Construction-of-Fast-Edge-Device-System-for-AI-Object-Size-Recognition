from flask import Flask, render_template, make_response, send_file, request, jsonify, redirect
####
import base64,io
# from flask_sqlalchemy import SQLAlchemy
# predict

import os, cv2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch,DefaultTrainer
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog
import tensormask
from tensormask import add_tensormask_config
import torch
from torch.utils.cpp_extension import CUDA_HOME 
#register
import register
####
from custom_visualizer import custom_Visualizer
####
import json
import shutil
# TRAIN
from detectron2.utils.logger import setup_logger
setup_logger()

#register
import register
#save json
import json
import numpy as np
import imantics
from imantics import Polygons, Mask

from PIL import Image

my_metadata = MetadataCatalog.get("my_train")

parameter = {}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:zhuantizhuanti@127.0.0.1:3306/db_python'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

@app.route('/')
def main_page():
    import os
    dir_path = './static/dataset/train/'
    # files = os.listdir(dir_path)
    # for filename in files:
    #     print(filename)
    #     img = cv2.imread(dir_path + filename)
    #     image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    #     cv2.imwrite(dir_path + filename, image)
    
    all_file_name = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, file_name) for file_name in all_file_name]

    widths=[]
    heights=[]

    for file_path in file_paths:
        img = Image.open(file_path)
        w,h=img.size
        widths.append(w)
        heights.append(h)

    predict_path = './static/pic/predict/'
    all_predict_name = os.listdir(predict_path)
    predict_file_paths = [os.path.join(predict_path, file_name) for file_name in all_predict_name]
    # images = ["../static/pic/picc.jpg", "../static/pic/piccc.jpg", "../static/pic/picc.jpg"]
    return render_template('temp.html', images=file_paths, predict_images=predict_file_paths, widths=widths, heights=heights)

@app.route('/qq')
def main_pagqe():
    return render_template('main_page.html')

@app.route('/parameter')
def parameter_paqe():
    return render_template('parameter.html')

@app.route('/train')
def train():
    source_path = "C:\\Users\\Yi Wei Chan\\Downloads\\annotations.json"
    destination_path = "C:\\Users\\Yi Wei Chan\\detectron2-main\\projects\\TensorMask\\database\\static\\dataset\\annotations\\train_annotations.coco.json"
    shutil.move(source_path, destination_path)
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("configs/tensormask_R_50_FPN_1x.yaml")
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    ##############pretrain
    cfg.OUTPUT_DIR = "training_output/tmask_coco"
    if parameter["pretrained"] == "yes" :
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
    else :
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "pretrained.pth") 
    
    #cfg.SOLVER.CHECKPOINT_PERIOD = 400 # save model

    #cfg.SOLVER.WARMUP_ITERS = 200
    #cfg.SOLVER.STEPS = (200, 380)#　STEPS: (480000, 520000)
    cfg.SOLVER.IMS_PER_BATCH = int(parameter['batchSize'])#2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = float(parameter['learningRate'])#0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 100*int(parameter["epochs"])    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 1 # only has one class (ballon)
   
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000 #每張圖片可以偵測的物件最大值

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodel.pth"))
    return render_template('parameter.html')


@app.route('/predict')
def predict():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    add_tensormask_config(cfg)
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.merge_from_file("configs/tensormask_R_50_FPN_1x.yaml")

    cfg.OUTPUT_DIR = "training_output/tmask_coco"

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.WEIGHTS = "C:\\Users\\Yi Wei Chan\\detectron2-main\\projects\\TensorMask\\database\\training_output\\tmask_coco\\mymodel.pth"
    #print("parameter")
    #print(parameter)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 1

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.TENSOR_MASK.NMS_THRESH_TEST = 0.75
    cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.075

    predictor = DefaultPredictor(cfg)
    #### show predict
    dataset_test = DatasetCatalog.get("my_test")
    for d in dataset_test:
        img = cv2.imread(d["file_name"])
        ####get the image file name
        img_name = d["file_name"].split("\\")
        img_name = img_name[len(img_name)-1]
        outputs = predictor(img)
        ###
        #v = Visualizer(img[:], metadata=my_metadata, scale=1.5)
        v = custom_Visualizer(img[:], metadata=my_metadata, scale=1.5)
        ########draw area
        area_list = []
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
            area_list.append(area)
        ########
        out = v.custom_draw(outputs["instances"].to("cpu"),area_list)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ####visualize prediction
        
        #v.save(dir_path)
        path = "./static/pic/predict/"
        cv2.imwrite(path + 'predict_' + img_name, out.get_image()[:, :, ::-1])
        #cv2.imshow("image",out.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        #break
    # return render_template('temp.html')
    return redirect('/')


# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100))
#     email = db.Column(db.String(100))
#
#     def __init__(self, name, email):
#         self.name = name
#         self.email = email

@app.route('/python-endpoint', methods=['POST'])
def handle_post_request():
    json_data = request.json
    print(json_data)
    print(type(json_data))

    parameter['learningRate'] = json_data ["learningRate"]
    parameter['batchSize'] = json_data ["batchSize"]
    parameter['epochs'] = json_data ["epochs"]
    parameter['pretrained'] = json_data ["pretrained"]
    print(parameter['pretrained'])
    return jsonify({'message': '收到'})

@app.route('/savejs')
def savejs():
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
            "name": register.CLASS_NAMES[0],    ####get class name ####12/11改 "name": register.CLASS_NAMES (原程式碼)
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
            "height": img.shape[0],
            "width": img.shape[1],
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
    with open("output.coco.json", "w") as f:
        json.dump(cocoData, f, indent = 4)
    return redirect('/')
    # return render_template('temp.html')

if __name__ == "__main__":
    #with app.app_context():
        # db.create_all()
    app.run(host='0.0.0.0', port=5000)
