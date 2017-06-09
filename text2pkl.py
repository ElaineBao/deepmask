#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : byx
# 2017-05-15 15:26

import cPickle
import sys
import numpy as np

def text2list(test_path):
    '''
    :param test_path: each line has bboxes of a image:
     x0,y0,x1,y1,score,...., (num of bboxes = line elements / 5)
    :return: bboxes_list of all images
    '''
    img_list=list()
    with open(test_path,'r') as f:
        for line_idx, line in enumerate(f.readlines()):
            elems = line.split(' ')
            bbox_num = len(elems) / 5
            if bbox_num == 0:
                print("line {}: num of bboxes: 0".format(line_idx))
                img_list.append(np.zeros((1,5)))
            elif bbox_num == 1:
                print("line {}: num of bboxes: {}".format(line_idx, bbox_num))
                x1 = int(elems[0])
                y1 = int(elems[1])
                x2 = int(elems[2])
                y2 = int(elems[3])
                score = float(elems[4])
                bboxes_array = np.array([[x1, y1, x2, y2, score]])
                img_list.append(bboxes_array)
            else:
                bboxes_array = np.array([])
                print("line {}: num of bboxes: {}".format(line_idx, bbox_num))
                for bbox_idx in xrange(bbox_num):
                    x1 = int(elems[bbox_idx * 5])
                    y1 = int(elems[bbox_idx * 5 + 1])
                    x2 = int(elems[bbox_idx * 5 + 2])
                    y2 = int(elems[bbox_idx * 5 + 3])
                    score = float(elems[bbox_idx * 5 + 4])
                    bbox = np.array([x1,y1,x2,y2,score])
                    if bboxes_array.size == 0:
                        bboxes_array = bbox
                    else:
                        bboxes_array = np.vstack((bboxes_array, bbox))
                print("---------------------------bboxes_array.shape:", bboxes_array.shape[0],bboxes_array.shape[1])
                img_list.append(bboxes_array)
    return img_list


def list2pkl(list, pkl_path):
    print('writing region proposal results to {}'.format(pkl_path))
    with open(pkl_path, 'wb') as fid:
        cPickle.dump(list, fid, cPickle.HIGHEST_PROTOCOL)
    print('done.')



if __name__ == '__main__':
    test_path = sys.argv[1]
    pkl_path = sys.argv[2]

    img_list =text2list(test_path)
    list2pkl(img_list, pkl_path)