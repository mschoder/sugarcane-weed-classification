# Script to train all FasterRCNN models on weed dataset varying base model and augmentations

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, random, copy
import matplotlib.pyplot as plt
import cv2
import time
import datetime
import logging

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from fvcore.transforms.transform import NoOpTransform

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm

setup_logger()

# build custom dataset and register it
def get_weed_dicts(img_dir, file_list):
    json_file = os.path.join(img_dir, "_weed_labels.json")
    with open(json_file) as f:
        img_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(img_anns.values()):
        if v["filename"] in file_list:  # train or test data
            record = {}
            filename = os.path.join(img_dir, v["filename"])
            # print(filename)
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annos = v["regions"]   # List of object attributes
            objs = []
            for anno in annos:
                if anno["region_attributes"]["label"] == "weed":
                    sa = anno["shape_attributes"]
                    obj = {
                        "bbox": [sa['x'], sa['y'], sa['width'], sa['height']],
                        "bbox_mode": BoxMode.XYWH_ABS, # or XYXY_ABS
                        "category_id": 0 if anno["region_attributes"]["label"] == "weed" else 1
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


# Create and register datasets
img_dir = '/home/mschoder/data/allweeds_600x400/'

json_file = os.path.join(img_dir, "_weed_labels.json")
print(json_file)
with open(json_file) as f:
    img_anns = json.load(f)

file_list = sorted([k for k in img_anns.keys()])
print("Number of images: ", len(file_list))
np.random.seed(31)
np.random.shuffle(file_list)
val_pct = 0.2  # 60-20-20 split
file_lists = {
    "val": file_list[:int(val_pct * len(file_list))],
    "test": file_list[int(val_pct * len(file_list)):int(2*val_pct * len(file_list))],
    "train": file_list[int(2*val_pct * len(file_list)):]
}

DatasetCatalog.clear()
for d in ["train", "val", "test"]:
    DatasetCatalog.register("weeds_" + d, lambda d=d: get_weed_dicts(img_dir, file_lists[d]))
    MetadataCatalog.get("weeds_" + d).set(thing_classes=["weed"])
weeds_metadata = MetadataCatalog.get("weeds_train")

# Build full dict
print('Creating datasets...')
# print(file_lists)
dataset_dicts = get_weed_dicts(img_dir, file_lists['train'])
print('Datasets created')


######## Define transformations ############
class HSV_EQ_Augmentation(T.Augmentation):
    def __init__(self):
        super().__init__()

    def get_transform(self, image):
        return HSV_EQ_Transform()

class HSV_EQ_Transform(T.Transform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self,img):
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_HSV)
        cl2 = clahe.apply(v)
        img_HSV_new = cv2.merge((h,s,cl2))
        img_BGR = cv2.cvtColor(img_HSV_new,cv2.COLOR_HSV2BGR)
        return img_BGR

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation

class HLS_EQ_Augmentation(T.Augmentation):
    def __init__(self):
        super().__init__()

    def get_transform(self, image):
        return HLS_EQ_Transform()

class HLS_EQ_Transform(T.Transform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self,img):
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(img_HLS)
        cl2 = clahe.apply(l)
        img_HLS_new = cv2.merge((h,cl2,s))
        img_BGR = cv2.cvtColor(img_HLS_new,cv2.COLOR_HLS2BGR)
        return img_BGR

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation

class NDI_CIVE_ExG_Augmentation(T.Augmentation):
    def __init__(self):
        super().__init__()

    def get_transform(self, image):
        return NDI_CIVE_ExG_Transform()

class NDI_CIVE_ExG_Transform(T.Transform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self,img):
        B,G,R = cv2.split(img)
        B = B.astype('float32');
        G = G.astype('float32')
        R = R.astype('float32')
        #NDI
        NDI=128.0*(G-R)/(G+R+1)
        #CIVE
        CIVE = 0.441*R-.881*G+.385*B+18.78745
        #ExG
        R_st = R/255; G_st = G/255 ; B_st = B/255
        tot =  R_st+G_st+B_st
        r = R_st/(tot+.01); g = G_st/(tot+.01); b = B_st/(tot+.01)
        ExG = 2*g-r-b
        img_out = cv2.merge((NDI,CIVE,ExG))
        return img_out

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation

class ExG_Augmentation(T.Augmentation):
    def __init__(self):
        super().__init__()

    def get_transform(self, image):
        return ExG_Transform()

class ExG_Transform(T.Transform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self,img):
        B,G,R = cv2.split(img)
        #ExG
        R_st = R/255; G_st = G/255 ; B_st = B/255
        tot =  R_st+G_st+B_st
        r = R_st/tot; g = G_st/tot; b = B_st/tot
        ExG = 2*g-r-b
        return ExG

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


########### Define custom training utilities ################

## Define hook for validation loss
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg, mapper=custom_mapper_test)) 
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_train)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=custom_mapper_test)



############# TRAINING CONFIG ####################

pretrained_models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                     "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                     "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"]

# pretrained_models = ["COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
#                      "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"]

# color_tfs = [NoOpTransform(), HSV_EQ_Transform(), 
#              HLS_EQ_Transform()]
color_tfs = [NDI_CIVE_ExG_Transform()]

for ptmodel in pretrained_models:
    for tf in color_tfs:

        ### Re-define custom mappers based on desired transforms
        def custom_mapper_train(dataset_dict):
            # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            image = utils.read_image(dataset_dict["file_name"], format="RGB")
            transform_list = []
            # Rotate 90 degrees to landscape if image is in portrait format
            height, width, _ = image.shape
            if height > width:
                transform_list.append(T.RotationTransform)
            transform_list = [T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]
            transform_list.append(tf)
            image, transforms = T.apply_transform_gens(transform_list, image)
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            return dataset_dict

        def custom_mapper_test(dataset_dict):
            dataset_dict = copy.deepcopy(dataset_dict) 
            image = utils.read_image(dataset_dict["file_name"], format="RGB")
            transform_list = []
            height, width, _ = image.shape
            if height > width:
                transform_list.append(T.RotationTransform)
            transform_list.append(tf)
            image, transforms = T.apply_transform_gens(transform_list, image)
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            return dataset_dict

        ### Select base model & initialized weights
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(ptmodel))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ptmodel) 

        cfg.DATASETS.TRAIN = ("weeds_train",)
        cfg.DATASETS.TEST = ("weeds_val",)

        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025  # TODO - tweak
        cfg.SOLVER.MAX_ITER = 750   # Adjusted based on validation mAP (500-1500)

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class (weed), other class is background

        cfg.TEST.EVAL_PERIOD = 100  # every 5 epochs

        cfg.OUTPUT_DIR = '/home/mschoder/experiment_outputs/' + \
                        'model_' + str(ptmodel) + '_' + \
                        'lr_' + str(cfg.SOLVER.BASE_LR) + '_' + \
                        'iters_' + str(cfg.SOLVER.MAX_ITER) + '_' + \
                        'tf_' + str(tf) + '/'

        # Set everything up to train, register hooks
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = CocoTrainer(cfg)
        val_loss = ValidationLoss(cfg)
        trainer.register_hooks([val_loss])
        # swap the order of PeriodicWriter and ValidationLoss
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
        trainer.resume_or_load(resume=False)

        ### TRAIN! ###
        print('Starting training for config: ', cfg.OUTPUT_DIR)
        trainer.train()
        print('Training succeeded for config: ', cfg.OUTPUT_DIR)
