#!/usr/bin/env python3
# Code are based on
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# Copyright (c) Bharath Hariharan.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np


def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):

    # # first load gt
    # if not os.path.isdir(cachedir):
    #     os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, "annots.pkl")
    # # read list of images
    # with open(imagesetfile, "r") as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]

    # if not os.path.isfile(cachefile):
    #     # load annots
    #     recs = {}
    #     for i, imagename in enumerate(imagenames):
    #         recs[imagename] = parse_rec(annopath.format(imagename)) #store xml file into recs{}, keys being image name and values being xml contents stored as key value pairs in a list
    #         if i % 100 == 0:
    #             print(f"Reading annotation for {i + 1}/{len(imagenames)}")


    #     class_recs = {}
    #     npos = 0
    #     for imagename in imagenames:
    #         R = [obj for obj in recs[imagename] if obj["name"] == classname] #store key value pair contents(of annotation.xml) into R if annotation file contain required class
    #         bbox = np.array([x["bbox"] for x in R]) #array of ground truth bounding boxes of all val images if they belong to required class (classname)
    #         difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
    #         det = [False] * len(R)
    #         npos = npos + sum(~difficult)   #npos = number of positive samples/target objects in val images for the specified classname(# of bounding boxes)
    #         class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    ##reading our predicted bounding boxes on val images for specified classname
################################################################################################################
    import json
    if classname == 'logo1':
        classs = 0
    else:
        classs = 1
    src_path = '/home/ec2-user/workspace_ahmad/YOLOX/datasets/custom_dataset/output.manifest'
    # Open the .manifest file and save its contents in a list
    data_list = list()
    for line in open(src_path):
        data_list.append(json.loads(line))
        
    res = np.empty((0, 4))
    BBGT = dict()
    npos = 0
    class_recs = {}
    for i, line in enumerate(data_list):
        key = line['source-ref'].split('/')[3]   #this returns the image names
    #     print(key)
        annotation = line['zegar-labeling-test']['annotations']
        if not annotation:
            BBGT[key] = {'bbox':res, 'difficult':np.zeros(len(res)).astype(bool), "det":np.zeros(len(res)).astype(bool)}
        else:    
            for line in annotation:
                class_id = line['class_id']
                xmin, ymin, obj_height, obj_width = float(line['left'])-1, float(line['top'])-1,float(line['height']), float(line['width'])
                xmax = xmin + obj_width
                ymax = ymin + obj_height
                dimensions = [xmin, ymin, xmax, ymax]
                if class_id ==classs:
                    res = np.vstack((res, dimensions))
                    
            BBGT[key] = {'bbox':res, 'difficult':np.zeros(len(res)).astype(bool),"det":np.zeros(len(res)).astype(bool)}
            npos= npos + len(res)
            res = np.empty((0, 4)) 
    class_recs = BBGT
#############################################################################################################

    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)  #nd = number of detections (for each class per image, as voc_eval runs for each class)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):  # run the loop for each bounding box(bb) of our specified class present in current image and compute iou with all BBGT present in that image.  
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
            )

            overlaps = inters / uni  #determine intersection/union of prediction with ground truth
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:  #check to see if max iou of this particular bb with BBGTs in this image is > threshold mAP
            if not R["difficult"][jmax]:  #Check to see if object is difficult or not at the given jmax index(prediction index at which iou is max)
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1  #Mark R(detected) at this index as 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
