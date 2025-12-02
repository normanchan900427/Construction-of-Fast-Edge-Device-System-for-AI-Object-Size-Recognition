# MY DATASET-*+
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
import os, cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

CLASS_NAMES = ["stone"]

DATASET_ROOT = r'C:\Users\Yi Wei Chan\detectron2-main\projects\TensorMask\database\static\dataset'

ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train_annotations.coco.json')

TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TEST_JSON = os.path.join(ANN_ROOT, 'test_annotations.coco.json')
#register_coco_instances("unlabeled_dataset", {}, "path/to/unlabeled_annotation.json", "path/to/unlabeled_images_directory")
register_coco_instances(
    name = "my_train",
    metadata = {},
    json_file = TRAIN_JSON,
    image_root = TRAIN_PATH
)

metadata = MetadataCatalog.get("my_train").set(
    thing_classes = CLASS_NAMES,
    evaluator_type = 'coco',
    json_file = TRAIN_JSON,
    image_root = TRAIN_PATH
)

register_coco_instances(
    name = "my_test",
    metadata = {},
    json_file = TEST_JSON,
    image_root = TEST_PATH
)

metadata = MetadataCatalog.get("my_test").set(
    thing_classes = CLASS_NAMES,
    evaluator_type = 'coco',
    json_file = TEST_JSON,
    image_root = TEST_PATH
)

################ test registion
#print([data_set
#    for data_set
#    in MetadataCatalog.list()
#])
#################

################# test json
#dataset_train = DatasetCatalog.get("my_train")

#for d in dataset_train:
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:], metadata=metadata, scale=2)
#    out = visualizer.draw_dataset_dict(d)
#    cv2.imshow("image",out.get_image()[:])
#    cv2.waitKey(0) 
#################
