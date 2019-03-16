
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import json
import sys
import cv2
import skimage
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from IPython import embed
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torchvision import transforms as T
from glob import glob
assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

# threshold for class score
threshold = 0.3
results_file = open("./submit/larger%s.csv"%str(threshold),"w")

if not os.path.exists("./submit/"):
    os.mkdir("./submit/")

if not os.path.exists("./outputs/"):
    os.mkdir("./outputs/")
if not os.path.exists("./best_models/"):
    os.mkdir("./best_models/")

def draw_caption(image, box, caption):
    
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def demo(image_lists):
    classes_name = ["1","2","3","4","5"]
    model = "/home/cdx/detect_strict/logs/199.pth"
    retinanet = torch.load(model)
    retinanet = retinanet.cuda()
    retinanet.eval()
    #detect
    transforms = T.Compose([
        Normalizer(),
        Resizer()
        ])
    result = []
    for filename in image_lists:
        file_dict = {}
        file_dict["filename"] = filename.split('/')[-1]
        image = skimage.io.imread(filename)
        sampler = {"img":image.astype(np.float32)/255.0,"annot":np.empty(shape=(5,5))}
        image_tf = transforms(sampler)
        scale = image_tf["scale"]
        new_shape = image_tf['img'].shape
        x = torch.autograd.Variable(image_tf['img'].unsqueeze(0).transpose(1,3), volatile=True)
        with torch.no_grad():
            scores,classes,bboxes = retinanet(x.cuda().float())
            bboxes /= scale
            scores = scores.cpu().data.numpy()
            bboxes = bboxes.cpu().data.numpy()
            # select threshold
            idxs = np.where(scores > threshold)[0]
            scores = scores[idxs]
            classes = classes[idxs]
            bboxes = bboxes[idxs]
            rects = []
            #embed()
            for i,box in enumerate(bboxes):
                box_dict = {}
                #box_dict["xmin"] = int(box[1])
                #box_dict["xmax"] = int(box[3])
                #box_dict["ymin"] = int(box[0])
                #box_dict["ymax"] = int(box[2])
                #box_dict["label"] = int(classes[i]) + 1
                #box_dict["confidence"] = float(scores[i])
                x_min = int(box[1])
                y_min = int(box[0])
                x_max = int(box[3])
                y_max = int(box[2])
                label = int(classes[i]) +1
                confidence = float(scores[i])
                box_dict["x_min"] = x_min
                box_dict["y_min"] = y_min
                box_dict["x_max"] = x_max
                box_dict["y_max"] = y_max
                box_dict["label"] = label
                box_dict["confidence"] = confidence
                rects.append(box_dict)
                draw_caption(image,[x_min,y_min,x_max,y_max],classes_name[int(classes[i])])
                cv2.rectangle(image,(int(x_min), int(y_min)), (int(x_max), int(y_max)),color=(0,0,255),thickness=2 )
                #results_file.write(filename.split("/")[-1] +","+ str(int(box[1])) + " " + str(int(box[0])) +  " " + str(int(box[3])) + " " +str(int(box[2])) + ","+ str(float(classes[i]) + 1) +"\n")
            
            print("Predicting image: %s "%filename)
            file_dict["rects"] = rects
            result.append(file_dict)
            cv2.imwrite("./outputs/%s"%filename.split("/")[-1],image)
    result_dict = {"results":result}
    with open('./submit/result.json','w') as f:
        json.dump(result_dict,f)
if __name__ == "__main__":
    root = "./data/test/"
    image_lists = glob(root+"*.jpg")
    demo(image_lists)
