# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:19:18 2020

@author: OWNER
"""

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader

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
        
        #NDI
        NDI=128*((G-R)/(G+R)+1)
        #CIVE
        CIVE = 0.441*R-.881*G+.385*B+18.78745
        #ExG
        R_st = R/255; G_st = G/255 ; B_st = B/255
        tot =  R_st+G_st+B_st
        r = R_st/tot; g = G_st/tot; b = B_st/tot
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


def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    transform_list = []
    # Rotate 90 degrees to landscape if image is in portrait format
    height, width, _ = image.shape
    if height > width:
        transform_list.append(T.RotationTransform)
    transform_list = [
                    #   T.RandomCrop('relative_range', (0.7, 0.7)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      HLS_EQ_Augmentation()
                    #   T.RandomBrightness(0.70, 1.3),
                    #   T.RandomContrast(0.7, 1.3),
                    #   T.RandomSaturation(0.7, 1.3),
                    #   T.ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style='choice')
                      ]
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


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)